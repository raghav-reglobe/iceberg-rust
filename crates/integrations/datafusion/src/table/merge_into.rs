// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Planning for merge-on-read MERGE INTO: classifies the ON condition and
//! WHEN clauses against the target/source schemas, derives the NARROW target
//! projection (only the target columns the expressions reference, plus
//! `_file`/`_pos`), binds everything to physical expressions, and assembles
//! the MoR execution chain.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::catalog::Session;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{
    Column, DFSchema, DFSchemaRef, DataFusionError, Result as DFResult, TableReference,
};
use datafusion::logical_expr::dml::{MergeIntoAction, MergeIntoClause};
use datafusion::logical_expr::utils::split_conjunction;
use datafusion::logical_expr::{Expr, Operator};
use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion::physical_plan::{ExecutionPlan, PhysicalExpr};
use iceberg::Catalog;
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::table::Table;

use crate::physical_plan::expr_to_predicate::convert_filters_to_predicate;
use crate::physical_plan::merge_mor::{
    IcebergMorMergeCommitExec, IcebergMorMergeExec, IcebergMorMergeWriteExec,
    IcebergMorTargetScanExec, MorActionPlan, MorClausePlan,
};
use crate::to_datafusion_error;

/// Which side of the merge a column (or a whole expression) references.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    Target,
    Source,
}

/// Build the MoR MERGE execution chain for `table`.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn build_mor_merge_plan(
    catalog: Arc<dyn Catalog>,
    table: Table,
    state: &dyn Session,
    source: Arc<dyn ExecutionPlan>,
    source_schema: DFSchemaRef,
    target_ref: TableReference,
    on: Expr,
    clauses: Vec<MergeIntoClause>,
) -> DFResult<Arc<dyn ExecutionPlan>> {
    if clauses.is_empty() {
        return Err(DataFusionError::Plan(
            "MERGE INTO requires at least one WHEN clause".to_string(),
        ));
    }
    let snapshot_id = table.metadata().current_snapshot_id();
    let table_arrow =
        schema_to_arrow_schema(table.metadata().current_schema()).map_err(to_datafusion_error)?;
    let target_names: HashSet<String> = table_arrow
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    let classify = |c: &Column| -> DFResult<Side> {
        column_side(c, &target_ref, &target_names, &source_schema)
    };

    // Collect every target column the merge expressions reference — that set
    // (plus _file/_pos) is the narrow target-scan projection.
    let mut referenced: HashSet<String> = HashSet::new();
    let mut collect = |expr: &Expr| -> DFResult<()> {
        expr.apply(|e| {
            if let Expr::Column(c) = e
                && classify(c)? == Side::Target
            {
                referenced.insert(c.name.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .map(|_| ())
    };
    collect(&on)?;
    for clause in &clauses {
        if let Some(p) = &clause.predicate {
            collect(p)?;
        }
        match &clause.action {
            MergeIntoAction::Update(assignments) => {
                for (_, value) in assignments {
                    collect(value)?;
                }
            }
            MergeIntoAction::Insert { values, .. } => {
                for value in values {
                    collect(value)?;
                }
            }
            MergeIntoAction::Delete => {}
        }
    }

    // Windowed merge: widen the narrow projection with the caller-named
    // extra columns so the HELD current-set can also serve the USING-side
    // self-read (the demote-union t2), whose projection exceeds the merge
    // expressions' own target references.
    if let Some(options) = state
        .config()
        .get_extension::<crate::physical_plan::MorMergeOptions>()
        && let Some(w) = options.window.as_ref()
        && w.hold_requested()
    {
        for col in w.held_extra_columns() {
            if target_names.contains(col) {
                referenced.insert(col.clone());
            } else {
                return Err(DataFusionError::Plan(format!(
                    "held_extra_columns: `{col}` is not a column of the merge target"
                )));
            }
        }
    }

    // Narrow projection in table-schema order.
    let mut narrow_names: Vec<String> = Vec::new();
    let mut narrow_fields: Vec<Field> = Vec::new();
    for field in table_arrow.fields() {
        if referenced.contains(field.name()) {
            narrow_names.push(field.name().clone());
            narrow_fields.push(field.as_ref().clone());
        }
    }

    // Schemas for physical-expression binding. Join output = source ++ target,
    // so the combined schema is built in that order.
    let narrow_arrow = {
        let mut fields = narrow_fields.clone();
        fields.push(Field::new(
            iceberg::metadata_columns::RESERVED_COL_NAME_FILE,
            datafusion::arrow::datatypes::DataType::Utf8,
            false,
        ));
        fields.push(Field::new(
            iceberg::metadata_columns::RESERVED_COL_NAME_POS,
            datafusion::arrow::datatypes::DataType::Int64,
            false,
        ));
        datafusion::arrow::datatypes::Schema::new(fields)
    };
    let target_df = DFSchema::try_from_qualified_schema(target_ref.clone(), &narrow_arrow)?;
    let combined_df = source_schema.join(&target_df)?;

    // ON condition: a conjunction of source-side = target-side equalities,
    // plus optional TARGET-ONLY residual conjuncts (e.g. the SCD2 demote
    // prune `t._is_current = true`). A residual filters the target scan —
    // pushed as an iceberg predicate for partition/file pruning AND
    // re-applied row-exactly on the scan output, so MERGE semantics hold
    // even where the iceberg pushdown is inexact.
    let mut join_on: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> = Vec::new();
    let mut residuals: Vec<Expr> = Vec::new();
    for conjunct in split_conjunction(&on) {
        let unaliased = unalias_ref(conjunct);
        // Target-only (or column-free) conjunct -> target scan residual.
        let mut only_target = true;
        for col in unaliased.column_refs() {
            if classify(col)? != Side::Target {
                only_target = false;
                break;
            }
        }
        if only_target {
            residuals.push(strip_qualifiers(unaliased.clone()));
            continue;
        }
        let Expr::BinaryExpr(binary) = unaliased else {
            return Err(DataFusionError::NotImplemented(format!(
                "MERGE ON condition must be a conjunction of equalities, got: {conjunct}"
            )));
        };
        if binary.op != Operator::Eq {
            return Err(DataFusionError::NotImplemented(format!(
                "MERGE ON condition supports only equality comparisons, got: {conjunct}"
            )));
        }
        let left_side = expr_side(&binary.left, &classify)?;
        let right_side = expr_side(&binary.right, &classify)?;
        let (source_expr, target_expr) = match (left_side, right_side) {
            (Side::Source, Side::Target) => (&binary.left, &binary.right),
            (Side::Target, Side::Source) => (&binary.right, &binary.left),
            _ => {
                return Err(DataFusionError::NotImplemented(format!(
                    "each MERGE ON equality must compare a source expression \
                     with a target expression, got: {conjunct}"
                )));
            }
        };
        join_on.push((
            state.create_physical_expr((**source_expr).clone(), &source_schema)?,
            state.create_physical_expr((**target_expr).clone(), &target_df)?,
        ));
    }
    if join_on.is_empty() {
        return Err(DataFusionError::NotImplemented(
            "MERGE ON condition requires at least one equality".to_string(),
        ));
    }

    // Bind the residuals for row-exact filtering (against the UNQUALIFIED
    // narrow scan schema) and convert them to an iceberg predicate for
    // scan pruning (best-effort; the row filter is what guarantees
    // semantics).
    let (target_residual, target_predicate) = if residuals.is_empty() {
        (None, None)
    } else {
        let combined = residuals
            .iter()
            .cloned()
            .reduce(|a, b| a.and(b))
            .expect("non-empty residuals");
        let narrow_unqualified = DFSchema::try_from(narrow_arrow.clone())?;
        let bound = state.create_physical_expr(combined, &narrow_unqualified)?;
        let predicate = convert_filters_to_predicate(&narrow_arrow, &residuals);
        (Some(bound), predicate)
    };

    // Bind the WHEN clauses to the join output schema.
    let mut mor_clauses: Vec<MorClausePlan> = Vec::with_capacity(clauses.len());
    for clause in &clauses {
        let predicate = clause
            .predicate
            .as_ref()
            .map(|p| state.create_physical_expr(p.clone(), &combined_df))
            .transpose()?;
        let action = match &clause.action {
            MergeIntoAction::Update(assignments) => MorActionPlan::Update(
                assignments
                    .iter()
                    .map(|(name, value)| {
                        Ok((
                            name.clone(),
                            state.create_physical_expr(value.clone(), &combined_df)?,
                        ))
                    })
                    .collect::<DFResult<Vec<_>>>()?,
            ),
            MergeIntoAction::Insert { columns, values } => {
                // An empty column list is full-width positional (validated by
                // the SQL planner against the target column count).
                let names: Vec<String> = if columns.is_empty() {
                    table_arrow
                        .fields()
                        .iter()
                        .map(|f| f.name().clone())
                        .collect()
                } else {
                    columns.clone()
                };
                MorActionPlan::Insert(
                    names
                        .into_iter()
                        .zip(values.iter())
                        .map(|(name, value)| {
                            Ok((
                                name,
                                state.create_physical_expr(value.clone(), &combined_df)?,
                            ))
                        })
                        .collect::<DFResult<Vec<_>>>()?,
                )
            }
            MergeIntoAction::Delete => MorActionPlan::Delete,
        };
        mor_clauses.push(MorClausePlan {
            kind: clause.kind,
            predicate,
            action,
        });
    }
    let mor_clauses = Arc::new(mor_clauses);

    // Assemble the chain.
    let target_scan = Arc::new(IcebergMorTargetScanExec::new(
        table.clone(),
        snapshot_id,
        narrow_names,
        narrow_fields,
        target_predicate,
        target_residual,
    )) as Arc<dyn ExecutionPlan>;
    let merge = Arc::new(IcebergMorMergeExec::new(
        source,
        target_scan,
        join_on,
        Arc::clone(&mor_clauses),
    )) as Arc<dyn ExecutionPlan>;
    let write = Arc::new(IcebergMorMergeWriteExec::new(
        table.clone(),
        snapshot_id,
        merge,
        mor_clauses,
    )) as Arc<dyn ExecutionPlan>;
    let coalesce = Arc::new(CoalescePartitionsExec::new(write));
    Ok(Arc::new(IcebergMorMergeCommitExec::new(
        table,
        catalog,
        snapshot_id,
        coalesce,
    )))
}

/// Rewrite every column reference to its UNQUALIFIED form so a target-side
/// residual (`t._is_current = true`) binds against the unqualified narrow
/// scan schema and converts to an iceberg predicate by plain column name.
fn strip_qualifiers(expr: Expr) -> Expr {
    use datafusion::common::tree_node::{Transformed, TreeNode};
    expr.transform(|e| {
        Ok(match e {
            Expr::Column(c) if c.relation.is_some() => {
                Transformed::yes(Expr::Column(Column::new_unqualified(c.name)))
            }
            other => Transformed::no(other),
        })
    })
    .expect("infallible transform")
    .data
}

fn unalias_ref(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(alias) => unalias_ref(&alias.expr),
        other => other,
    }
}

/// Which side does `column` belong to? Qualified columns resolve by their
/// table reference; unqualified columns resolve by name, erroring when both
/// sides could satisfy them.
fn column_side(
    column: &Column,
    target_ref: &TableReference,
    target_names: &HashSet<String>,
    source_schema: &DFSchema,
) -> DFResult<Side> {
    if let Some(relation) = &column.relation {
        if relation == target_ref {
            return Ok(Side::Target);
        }
        if source_schema
            .field_with_qualified_name(relation, &column.name)
            .is_ok()
        {
            return Ok(Side::Source);
        }
        return Err(DataFusionError::Plan(format!(
            "MERGE expression references unknown relation '{relation}' (column '{}')",
            column.name
        )));
    }
    let in_target = target_names.contains(&column.name);
    let in_source = source_schema
        .field_with_unqualified_name(&column.name)
        .is_ok();
    match (in_target, in_source) {
        (true, false) => Ok(Side::Target),
        (false, true) => Ok(Side::Source),
        (true, true) => Err(DataFusionError::Plan(format!(
            "ambiguous MERGE column reference '{}': present in both target and source; \
             qualify it",
            column.name
        ))),
        (false, false) => Err(DataFusionError::Plan(format!(
            "unknown MERGE column reference '{}'",
            column.name
        ))),
    }
}

/// The single side an expression references; expressions mixing both sides
/// (or referencing neither) are not valid join-key operands.
fn expr_side(expr: &Expr, classify: &impl Fn(&Column) -> DFResult<Side>) -> DFResult<Side> {
    let mut side: Option<Side> = None;
    expr.apply(|e| {
        if let Expr::Column(c) = e {
            let s = classify(c)?;
            match side {
                None => side = Some(s),
                Some(prev) if prev != s => {
                    return Err(DataFusionError::NotImplemented(
                        "MERGE ON equality operands must not mix target and source columns"
                            .to_string(),
                    ));
                }
                _ => {}
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    side.ok_or_else(|| {
        DataFusionError::NotImplemented(
            "MERGE ON equality operands must reference a column".to_string(),
        )
    })
}
