#![allow(unsafe_op_in_unsafe_fn)]
mod filtered_bit_array;
mod l1_l2;

use std::cmp::min;
use std::ops::Range;

use filtered_bit_array::FilteredBitArray;
use l1_l2::*;
use polars_core::chunked_array::ChunkedArray;
use polars_core::datatypes::{IdxCa, NumericNative, PolarsNumericType};
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::{_set_partition_size, split, Container};
use polars_core::{POOL, with_match_physical_numeric_polars_type};
use polars_error::{PolarsResult, polars_err};
use polars_utils::IdxSize;
use polars_utils::binary_search::ExponentialSearch;
use polars_utils::itertools::Itertools;
use polars_utils::total_ord::{TotalEq, TotalOrd};
use rayon::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use arrow::pushable::Pushable;
use crate::frame::{JoinType, _finish_join};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InequalityOperator {
    #[default]
    Lt,
    LtEq,
    Gt,
    GtEq,
}

impl InequalityOperator {
    fn is_strict(&self) -> bool {
        matches!(self, InequalityOperator::Gt | InequalityOperator::Lt)
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct IEJoinOptions {
    pub join_type: JoinType,
    pub operator1: InequalityOperator,
    pub operator2: Option<InequalityOperator>,
}

#[allow(clippy::too_many_arguments)]
fn ie_join_impl_t<T: PolarsNumericType>(
    slice: Option<(i64, usize)>,
    l1_order: IdxCa,
    l2_order: &[IdxSize],
    op1: InequalityOperator,
    op2: InequalityOperator,
    x: Series,
    y_ordered_by_x: Series,
    left_height: usize,
) -> PolarsResult<(Vec<IdxSize>, Vec<IdxSize>)> {
    // Create a bit array with order corresponding to L1,
    // denoting which entries have been visited while traversing L2.
    let mut bit_array = FilteredBitArray::from_len_zeroed(l1_order.len());

    let mut left_row_idx: Vec<IdxSize> = vec![];
    let mut right_row_idx: Vec<IdxSize> = vec![];

    let slice_end = slice_end_index(slice);
    let mut match_count = 0;

    let ca: &ChunkedArray<T> = x.as_ref().as_ref();
    let l1_array = build_l1_array(ca, &l1_order, left_height as IdxSize)?;

    if op2.is_strict() {
        // For strict inequalities, we rely on using a stable sort of l2 so that
        // p values only increase as we traverse a run of equal y values.
        // To handle inclusive comparisons in x and duplicate x values we also need the
        // sort of l1 to be stable, so that the left hand side entries come before the right
        // hand side entries (as we mark visited entries from the right hand side).
        for &p in l2_order {
            match_count += unsafe {
                l1_array.process_entry(
                    p as usize,
                    &mut bit_array,
                    op1,
                    &mut left_row_idx,
                    &mut right_row_idx,
                )
            };

            if slice_end.is_some_and(|end| match_count >= end) {
                break;
            }
        }
    } else {
        let l2_array = build_l2_array(&y_ordered_by_x, l2_order)?;

        // For non-strict inequalities in l2, we need to track runs of equal y values and only
        // check for matches after we reach the end of the run and have marked all rhs entries
        // in the run as visited.
        let mut run_start = 0;

        for i in 0..l2_array.len() {
            // Elide bound checks
            unsafe {
                let item = l2_array.get_unchecked(i);
                let p = item.l1_index;
                l1_array.mark_visited(p as usize, &mut bit_array);

                if item.run_end {
                    for l2_item in l2_array.get_unchecked(run_start..i + 1) {
                        let p = l2_item.l1_index;
                        match_count += l1_array.process_lhs_entry(
                            p as usize,
                            &bit_array,
                            op1,
                            &mut left_row_idx,
                            &mut right_row_idx,
                        );
                    }

                    run_start = i + 1;

                    if slice_end.is_some_and(|end| match_count >= end) {
                        break;
                    }
                }
            }
        }
    }
    Ok((left_row_idx, right_row_idx))
}

fn piecewise_merge_join_impl_t<T, P>(
    slice: Option<(i64, usize)>,
    left_ordered: Series,
    right_ordered: Series,
    mut pred: P,
    join_dispatcher: &mut JoinDispatcher,
)
where
    T: PolarsNumericType,
    P: FnMut(&T::Native, &T::Native) -> bool,
{
    let slice_end = slice_end_index(slice);

    let left_ca: &ChunkedArray<T> = left_ordered.as_ref().as_ref();
    let right_ca: &ChunkedArray<T> = right_ordered.as_ref().as_ref();

    let mut left_idx = 0;
    let mut right_idx = 0;
    let mut match_count = 0;

    while left_idx < left_ca.len() {
        debug_assert!(left_ca.get(left_idx).is_some());
        let left_val = unsafe { left_ca.value_unchecked(left_idx) };
        while right_idx < right_ca.len() {
            debug_assert!(right_ca.get(right_idx).is_some());
            let right_val = unsafe { right_ca.value_unchecked(right_idx) };
            if pred(&left_val, &right_val) {
                // if this is the first match, then all right rows before are unmatched
                if match_count == 0 && right_idx > 0 {
                    let unmatched_right_row_idxs = 0..right_idx;
                    join_dispatcher.handle_unmatched_right(unmatched_right_row_idxs);
                }

                // If the predicate is true, then it will also be true for all
                // remaining rows from the right side.
                let right_end_idx = match slice_end {
                    None => right_ca.len(),
                    Some(end) => min(right_ca.len(), (end as usize) - match_count + right_idx),
                };
                let included_right_row_idxs = right_idx..right_end_idx;
                join_dispatcher.handle_left_matches_right(left_idx, included_right_row_idxs);
                match_count += right_end_idx - right_idx;
                break;
            } else {
                right_idx += 1;
            }
        }
        if right_idx == right_ca.len() {
            // We've reached the end of the right side
            // so there can be no more matches for LHS rows
            let left_end_idx = match slice_end {
                None => left_ca.len(),
                Some(end) => min(left_ca.len(), (end as usize) - match_count),
            };
            let unmatched_left_row_idxs = left_idx..left_end_idx;
            join_dispatcher.handle_unmatched_left(unmatched_left_row_idxs);
            break;
        }
        if slice_end.is_some_and(|end| match_count >= end as usize) {
            break;
        }
        left_idx += 1;
    }
}

pub(super) fn iejoin_par(
    left: &DataFrame,
    right: &DataFrame,
    selected_left: Vec<Series>,
    selected_right: Vec<Series>,
    options: &IEJoinOptions,
    suffix: Option<PlSmallStr>,
    slice: Option<(i64, usize)>,
) -> PolarsResult<DataFrame> {
    let l1_descending = matches!(
        options.operator1,
        InequalityOperator::Gt | InequalityOperator::GtEq
    );

    let l1_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(l1_descending);

    let sl = &selected_left[0];
    let l1_s_l = sl
        .arg_sort(l1_sort_options)
        .slice(sl.null_count() as i64, sl.len() - sl.null_count());

    let sr = &selected_right[0];
    let l1_s_r = sr
        .arg_sort(l1_sort_options)
        .slice(sr.null_count() as i64, sr.len() - sr.null_count());

    // Because we do a cartesian product, the number of partitions is squared.
    // We take the sqrt, but we don't expect every partition to produce results and work can be
    // imbalanced, so we multiply the number of partitions by 2, which leads to 2^2= 4
    let n_partitions = (_set_partition_size() as f32).sqrt() as usize * 2;
    let splitted_a = split(&l1_s_l, n_partitions);
    let splitted_b = split(&l1_s_r, n_partitions);

    let cartesian_prod = splitted_a
        .iter()
        .flat_map(|l| splitted_b.iter().map(move |r| (l, r)))
        .collect::<Vec<_>>();

    let iter = cartesian_prod.par_iter().map(|(l_l1_idx, r_l1_idx)| {
        if l_l1_idx.is_empty() || r_l1_idx.is_empty() {
            return Ok(None);
        }
        fn get_extrema<'a>(
            l1_idx: &'a IdxCa,
            s: &'a Series,
        ) -> Option<(AnyValue<'a>, AnyValue<'a>)> {
            let first = l1_idx.first()?;
            let last = l1_idx.last()?;

            let start = s.get(first as usize).unwrap();
            let end = s.get(last as usize).unwrap();

            Some(if start < end {
                (start, end)
            } else {
                (end, start)
            })
        }
        let Some((min_l, max_l)) = get_extrema(l_l1_idx, sl) else {
            return Ok(None);
        };
        let Some((min_r, max_r)) = get_extrema(r_l1_idx, sr) else {
            return Ok(None);
        };

        let include_block = match options.operator1 {
            InequalityOperator::Lt => min_l < max_r,
            InequalityOperator::LtEq => min_l <= max_r,
            InequalityOperator::Gt => max_l > min_r,
            InequalityOperator::GtEq => max_l >= min_r,
        };

        if include_block {
            let (mut l, mut r) = unsafe {
                (
                    selected_left
                        .iter()
                        .map(|s| s.take_unchecked(l_l1_idx))
                        .collect_vec(),
                    selected_right
                        .iter()
                        .map(|s| s.take_unchecked(r_l1_idx))
                        .collect_vec(),
                )
            };
            let sorted_flag = if l1_descending {
                IsSorted::Descending
            } else {
                IsSorted::Ascending
            };
            // We sorted using the first series
            l[0].set_sorted_flag(sorted_flag);
            r[0].set_sorted_flag(sorted_flag);

            // Compute the row indexes
            let (idx_l, idx_r) = if options.operator2.is_some() {
                iejoin_tuples(l, r, options, None)
            } else {
                piecewise_merge_join_tuples(l, r, options, None)
            }?;

            if idx_l.is_empty() {
                return Ok(None);
            }

            // These are row indexes in the slices we have given, so we use those to gather in the
            // original l1 offset arrays. This gives us indexes in the original tables.
            unsafe {
                Ok(Some((
                    l_l1_idx.take_unchecked(&idx_l),
                    r_l1_idx.take_unchecked(&idx_r),
                )))
            }
        } else {
            Ok(None)
        }
    });

    let row_indices = POOL.install(|| iter.collect::<PolarsResult<Vec<_>>>())?;

    let mut left_idx = IdxCa::default();
    let mut right_idx = IdxCa::default();
    for (l, r) in row_indices.into_iter().flatten() {
        left_idx.append(&l)?;
        right_idx.append(&r)?;
    }
    if let Some((offset, end)) = slice {
        left_idx = left_idx.slice(offset, end);
        right_idx = right_idx.slice(offset, end);
    }

    unsafe { materialize_join(left, right, &left_idx, &right_idx, suffix) }
}

pub(super) fn iejoin(
    left: &DataFrame,
    right: &DataFrame,
    selected_left: Vec<Series>,
    selected_right: Vec<Series>,
    options: &IEJoinOptions,
    suffix: Option<PlSmallStr>,
    slice: Option<(i64, usize)>,
) -> PolarsResult<DataFrame> {
    let (left_row_idx, right_row_idx) = if options.operator2.is_some() {
        iejoin_tuples(selected_left, selected_right, options, slice)
    } else {
        piecewise_merge_join_tuples(selected_left, selected_right, options, slice)
    }?;
    unsafe { materialize_join(left, right, &left_row_idx, &right_row_idx, suffix) }
}

unsafe fn materialize_join(
    left: &DataFrame,
    right: &DataFrame,
    left_row_idx: &IdxCa,
    right_row_idx: &IdxCa,
    suffix: Option<PlSmallStr>,
) -> PolarsResult<DataFrame> {
    try_raise_keyboard_interrupt();
    let (join_left, join_right) = {
        POOL.join(
            || left.take_unchecked(left_row_idx),
            || right.take_unchecked(right_row_idx),
        )
    };

    // right might be empty in case of a SEMI or ANTI join, then join_left is our result
    if right.len() == 0 {
        Ok(join_left)
    } else {
        _finish_join(join_left, join_right, suffix)
    }
}

/// Inequality join. Matches rows between two DataFrames using two inequality operators
/// (one of [<, <=, >, >=]).
/// Based on Khayyat et al. 2015, "Lightning Fast and Space Efficient Inequality Joins"
/// and extended to work with duplicate values.
fn iejoin_tuples(
    selected_left: Vec<Series>,
    selected_right: Vec<Series>,
    options: &IEJoinOptions,
    slice: Option<(i64, usize)>,
) -> PolarsResult<(IdxCa, IdxCa)> {
    if selected_left.len() != 2 {
        return Err(
            polars_err!(ComputeError: "IEJoin requires exactly two expressions from the left DataFrame"),
        );
    };
    if selected_right.len() != 2 {
        return Err(
            polars_err!(ComputeError: "IEJoin requires exactly two expressions from the right DataFrame"),
        );
    };

    let op1 = options.operator1;
    let op2 = match options.operator2 {
        None => {
            return Err(polars_err!(ComputeError: "IEJoin requires two inequality operators"));
        },
        Some(op2) => op2,
    };

    // Determine the sort order based on the comparison operators used.
    // We want to sort L1 so that "x[i] op1 x[j]" is true for j > i,
    // and L2 so that "y[i] op2 y[j]" is true for j < i
    // (except in the case of duplicates and strict inequalities).
    // Note that the algorithms published in Khayyat et al. have incorrect logic for
    // determining whether to sort descending.
    let l1_descending = matches!(op1, InequalityOperator::Gt | InequalityOperator::GtEq);
    let l2_descending = matches!(op2, InequalityOperator::Lt | InequalityOperator::LtEq);

    let mut x = selected_left[0].to_physical_repr().into_owned();
    let left_height = x.len();

    x.extend(&selected_right[0].to_physical_repr())?;
    // Rechunk because we will gather.
    let x = x.rechunk();

    let mut y = selected_left[1].to_physical_repr().into_owned();
    y.extend(&selected_right[1].to_physical_repr())?;
    // Rechunk because we will gather.
    let y = y.rechunk();

    let l1_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(l1_descending);
    // Get ordering of x, skipping any null entries as these cannot be matches
    let l1_order = x
        .arg_sort(l1_sort_options)
        .slice(x.null_count() as i64, x.len() - x.null_count());

    let y_ordered_by_x = unsafe { y.take_unchecked(&l1_order) };
    let l2_sort_options = SortOptions::default()
        .with_maintain_order(true)
        .with_nulls_last(false)
        .with_order_descending(l2_descending);
    // Get the indexes into l1, ordered by y values.
    // l2_order is the same as "p" from Khayyat et al.
    let l2_order = y_ordered_by_x.arg_sort(l2_sort_options).slice(
        y_ordered_by_x.null_count() as i64,
        y_ordered_by_x.len() - y_ordered_by_x.null_count(),
    );
    let l2_order = l2_order.rechunk();
    let l2_order = l2_order.downcast_as_array().values().as_slice();

    let (left_row_idx, right_row_idx) = with_match_physical_numeric_polars_type!(x.dtype(), |$T| {
         ie_join_impl_t::<$T>(
            slice,
            l1_order,
            l2_order,
            op1,
            op2,
            x,
            y_ordered_by_x,
            left_height
        )
    })?;

    debug_assert_eq!(left_row_idx.len(), right_row_idx.len());
    let left_row_idx = IdxCa::from_vec("".into(), left_row_idx);
    let right_row_idx = IdxCa::from_vec("".into(), right_row_idx);
    let (left_row_idx, right_row_idx) = match slice {
        None => (left_row_idx, right_row_idx),
        Some((offset, len)) => (
            left_row_idx.slice(offset, len),
            right_row_idx.slice(offset, len),
        ),
    };
    Ok((left_row_idx, right_row_idx))
}


/// A class that holds a JoinType value.
pub struct JoinDispatcher {
    join_type: JoinType,
    left_idxs: Vec<usize>,
    right_idxs: Vec<usize>,
}

impl JoinDispatcher {
    pub fn new(join_type: JoinType) -> Self {
        Self {
            join_type,
            left_idxs: vec![],
            right_idxs: vec![],
        }
    }

    pub fn handle_left_matches_right(&mut self, left_match_idx: usize, right_match_idxs: Range<usize>) {
        match self.join_type {
            JoinType::Left | JoinType::Inner | JoinType::Right | JoinType::Full =>
                right_match_idxs.for_each(|right_match_idx| {
                    self.left_idxs.push(left_match_idx);
                    self.right_idxs.push(right_match_idx);
                }),
            JoinType::Semi =>
                self.left_idxs.push(left_match_idx),
            _ => {}
        };
    }

    pub fn handle_unmatched_left(&mut self, left_unmatched_idxs: Range<usize>) {
        match self.join_type {
            JoinType::Left | JoinType::Full => {
                self.left_idxs.extend(left_unmatched_idxs.clone());
                self.right_idxs.extend_null_constant(left_unmatched_idxs.len())
            },
            JoinType::Anti => {
                self.left_idxs.extend(left_unmatched_idxs);
            },
            _ => {}
        };
    }

    pub fn handle_unmatched_right(&mut self, right_unmatched_idxs: Range<usize>) {
        match self.join_type {
            JoinType::Right | JoinType::Full => {
                self.left_idxs.extend_null_constant(right_unmatched_idxs.len());
                self.right_idxs.extend(right_unmatched_idxs)
            },
            _ => {}
        };
    }
}

/// Piecewise merge join, for joins with only a single inequality.
fn piecewise_merge_join_tuples(
    selected_left: Vec<Series>,
    selected_right: Vec<Series>,
    options: &IEJoinOptions,
    slice: Option<(i64, usize)>,
) -> PolarsResult<(IdxCa, IdxCa)> {
    if selected_left.len() != 1 {
        return Err(
            polars_err!(ComputeError: "Piecewise merge join requires exactly one expression from the left DataFrame"),
        );
    };
    if selected_right.len() != 1 {
        return Err(
            polars_err!(ComputeError: "Piecewise merge join requires exactly one expression from the right DataFrame"),
        );
    };
    if options.operator2.is_some() {
        return Err(
            polars_err!(ComputeError: "Piecewise merge join expects only one inequality operator"),
        );
    }

    let op = options.operator1;
    // The left side is sorted such that if the condition is false, it will also
    // be false for the same RHS row and all following LHS rows.
    // The right side is sorted such that if the condition is true then it is also
    // true for the same LHS row and all following RHS rows.
    // The desired sort order should match the l1 order used in iejoin_par
    // so we don't need to re-sort slices when doing a parallel join.
    let descending = matches!(op, InequalityOperator::Gt | InequalityOperator::GtEq);

    let left = selected_left[0].to_physical_repr().into_owned();
    let mut right = selected_right[0].to_physical_repr().into_owned();
    let must_cast = right.dtype().matches_schema_type(left.dtype())?;
    if must_cast {
        right = right.cast(left.dtype())?;
    }

    fn get_sorted(series: Series, descending: bool) -> (Series, Option<IdxCa>) {
        let expected_flag = if descending {
            IsSorted::Descending
        } else {
            IsSorted::Ascending
        };
        if (series.is_sorted_flag() == expected_flag || series.len() <= 1) && !series.has_nulls() {
            // Fast path, no need to re-sort
            (series, None)
        } else {
            let sort_options = SortOptions::default()
                .with_nulls_last(false)
                .with_order_descending(descending);

            // Get order and slice to ignore any null values, which cannot be match results
            let mut order = series.arg_sort(sort_options).slice(
                series.null_count() as i64,
                series.len() - series.null_count(),
            );
            order.rechunk_mut();
            let ordered = unsafe { series.take_unchecked(&order) };
            (ordered, Some(order))
        }
    }

    let (left_ordered, left_order) = get_sorted(left, descending);
    debug_assert!(
        left_order
            .as_ref()
            .is_none_or(|order| order.chunks().len() == 1)
    );
    let left_order = left_order
        .as_ref()
        .map(|order| order.downcast_get(0).unwrap().values().as_slice());

    let (right_ordered, right_order) = get_sorted(right, descending);
    debug_assert!(
        right_order
            .as_ref()
            .is_none_or(|order| order.chunks().len() == 1)
    );
    let right_order = right_order
        .as_ref()
        .map(|order| order.downcast_get(0).unwrap().values().as_slice());

    debug_assert!(left_order.is_none_or(|order| order.len() == left_ordered.len()));
    debug_assert!(right_order.is_none_or(|order| order.len() == right_ordered.len()));

    print!("join type {}", options.join_type);

    let mut join_dispatcher = JoinDispatcher::new(options.join_type.clone());

    with_match_physical_numeric_polars_type!(left_ordered.dtype(), |$T| {
        match op {
            InequalityOperator::Lt => piecewise_merge_join_impl_t::<$T, _>(
                slice,
                left_ordered,
                right_ordered,
                |l, r| l.tot_lt(r),
                &mut join_dispatcher,
            ),
            InequalityOperator::LtEq => piecewise_merge_join_impl_t::<$T, _>(
                slice,
                left_ordered,
                right_ordered,
                |l, r| l.tot_le(r),
                &mut join_dispatcher,
            ),
            InequalityOperator::Gt => piecewise_merge_join_impl_t::<$T, _>(
                slice,
                left_ordered,
                right_ordered,
                |l, r| l.tot_gt(r),
                &mut join_dispatcher,
            ),
            InequalityOperator::GtEq => piecewise_merge_join_impl_t::<$T, _>(
                slice,
                left_ordered,
                right_ordered,
                |l, r| l.tot_ge(r),
                &mut join_dispatcher,
            ),
        }
    });

    let (left_idxs, right_idxs) = (join_dispatcher.left_idxs, join_dispatcher.right_idxs);

    fn idxs_to_rows(idxs: Vec<usize>, order: Option<&[IdxSize]>) -> Vec<IdxSize> {
        match order {
            None => idxs.iter().map(|v| *v as IdxSize).collect(),
            Some(order) => idxs.into_iter().map(|v| order[v]).collect(),
        }
    }

    let left_row_idx = idxs_to_rows(left_idxs, left_order);
    let right_row_idx = idxs_to_rows(right_idxs, right_order);

    if options.join_type == JoinType::Semi || options.join_type == JoinType::Anti {
        debug_assert_eq!(right_row_idx.len(), 0);
    } else {
        debug_assert_eq!(left_row_idx.len(), right_row_idx.len());
    }

    let left_row_idx = IdxCa::from_vec("".into(), left_row_idx);
    let right_row_idx = IdxCa::from_vec("".into(), right_row_idx);
    let (left_row_idx, right_row_idx) = match slice {
        None => (left_row_idx, right_row_idx),
        Some((offset, len)) => (
            left_row_idx.slice(offset, len),
            right_row_idx.slice(offset, len),
        ),
    };
    Ok((left_row_idx, right_row_idx))
}

fn slice_end_index(slice: Option<(i64, usize)>) -> Option<i64> {
    match slice {
        Some((offset, len)) if offset >= 0 => Some(offset.saturating_add_unsigned(len as u64)),
        _ => None,
    }
}
