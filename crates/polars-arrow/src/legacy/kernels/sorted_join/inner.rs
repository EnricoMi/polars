use polars_utils::itertools::Itertools;
use super::*;

pub fn join<T: PartialOrd + Copy + Debug>(
    left: &[T],
    right: &[T],
    left_offset: IdxSize,
    consume: impl Fn(Vec<IdxSize>, Vec<IdxSize>) -> InnerJoinIds,
) -> InnerJoinIds {
    if left.is_empty() || right.is_empty() {
        return (vec![], vec![]);
    }

    // * 1.5 because of possible duplicates
    let cap = (std::cmp::min(left.len(), right.len()) as f32 * 1.5) as usize;
    let mut out_rhs = Vec::with_capacity(cap);
    let mut out_lhs = Vec::with_capacity(cap);

    let mut right_idx = 0 as IdxSize;
    // left array could start lower than right;
    // left: [-1, 0, 1, 2],
    // right: [1, 2, 3]
    let first_right = right[0];
    let mut left_idx = left.partition_point(|v| v < &first_right) as IdxSize;

    for &val_l in &left[left_idx as usize..] {
        let mut left_end = left_idx;
        while left.len() > left_end as usize && left[left_end as usize] == val_l {
            left_end += 1
        }

        while let Some(&val_r) = right.get(right_idx as usize) {
            // matching join key
            if val_l == val_r {
                let mut right_end = right_idx;
                while right.len() > right_end as usize && right[right_end as usize] == val_r {
                    right_end += 1
                }

                let left_idxs = (left_idx + left_offset..left_end + left_offset).collect_vec();
                let right_idxs = (right_idx..right_end).collect_vec();
                let (lhs, rhs) = consume(left_idxs, right_idxs);
                out_lhs.extend(lhs);
                out_rhs.extend(rhs);
                right_idx = right_end;
                break;
            }

            // right is larger than left.
            if val_r > val_l {
                break;
            }
            // continue looping the right side
            right_idx += 1;
        }
        left_idx = left_end;
        if left_idx as usize >= left.len() || right_idx as usize >= right.len() {
            break
        }
    }
    (out_lhs, out_rhs)
}

pub fn cartesian(
    left_idxs: Vec<IdxSize>,
    right_idxs: Vec<IdxSize>,
) -> InnerJoinIds {
    // this maintains left_right order
    let (left_len, right_len) = (left_idxs.len(), right_idxs.len());
    match (left_len, right_len) {
        (0, 0) => (vec![], vec![]),
        (1, 1) => (left_idxs, right_idxs),
        (1, _) => (left_idxs.repeat(right_len), right_idxs),
        (_, 1) => (left_idxs, right_idxs.repeat(left_len)),
        _ => {
            let mut out_lhs = Vec::with_capacity(left_len * right_len);
            for left_idx in left_idxs {
                out_lhs.extend(vec![left_idx; right_len]);
            }
            let out_rhs = right_idxs.repeat(left_len);
            (out_lhs, out_rhs)
        }
    }
}

#[cfg(test)]
mod test {
    use polars_utils::itertools::Itertools;
    use super::*;

    #[test]
    fn test_inner_join() {
        let lhs = &[0, 1, 1, 2, 3, 5];
        let rhs = &[0, 1, 1, 3, 4];

        let (l_idx, r_idx) = join(lhs, rhs, 0, cartesian);

        assert_eq!(&l_idx, &[0, 1, 1, 2, 2, 4]);
        assert_eq!(&r_idx, &[0, 1, 2, 1, 2, 3]);

        let lhs = &[4, 4, 4, 4, 5, 6, 6, 7, 7, 7];
        let rhs = &[0, 1, 2, 3, 4, 4, 4, 6, 7, 7];
        let (l_idx, r_idx) = join(lhs, rhs, 0, cartesian);

        assert_eq!(
            &l_idx,
            &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 7, 8, 8, 9, 9]
        );
        assert_eq!(
            &r_idx,
            &[4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 7, 8, 9, 8, 9, 8, 9]
        );
    }

    #[test]
    fn test_cartesian() {
        let N = 100;
        let M = 10;
        let left_idxs = (0..N);
        let right_idxs = (N..N + M);
        let (l_idxs, r_idxs) = cartesian(left_idxs.collect_vec(), right_idxs.collect_vec());
        let mut expected_l_idxs = (0..N).collect_vec().repeat(M as usize);
        expected_l_idxs.sort();
        let expected_r_idxs = (N..N + M).collect_vec().repeat(N as usize);
        assert_eq!(l_idxs, expected_l_idxs);
        assert_eq!(r_idxs, expected_r_idxs);
    }
}
