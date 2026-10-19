//! How a target spreads one tile over its execution units.
//!
//! A tile is a logical unit: the kernel says which elements it touches, not
//! which thread touches them. That is the freedom this module spends. A target
//! reports what its hardware allows, and the lowering carries every per-element
//! loop across the workgroup instead of running it on one invocation.

/// What a target's execution units allow, in the terms the lowering needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchShape {
    /// Invocations that share one tile, along each axis.
    ///
    /// `[0]` is the row axis of a tile, `[1]` its column axis. Both are one on
    /// a target that runs a whole tile on one invocation.
    pub threads: [u32; 2],
    /// Invocations a barrier round can wait for.
    ///
    /// A workgroup is the unit a barrier synchronizes, so the product of
    /// `threads` may not exceed this.
    pub max_invocations: u32,
}

impl LaunchShape {
    /// One invocation to a tile, which is what a scalar target does.
    pub const SERIAL: Self = Self {
        threads: [1, 1],
        max_invocations: 1,
    };

    /// How many invocations this shape gives one tile.
    pub fn count(&self) -> u32 {
        self.threads[0].saturating_mul(self.threads[1])
    }

    /// True when the tile's loops run across more than one invocation.
    pub fn is_parallel(&self) -> bool {
        self.count() > 1
    }

    /// The shape that spreads one `rows` by `cols` tile over `max_invocations`,
    /// never asking for more threads than the tile has elements.
    ///
    /// The row axis is filled first. A tile's rows are the independent ones in
    /// every shape the lowering emits: a pointwise store, a reduce along the
    /// last axis and a matmul's accumulator all keep a row to itself, so
    /// splitting rows costs one barrier where splitting columns costs one per
    /// reduction.
    pub fn for_tile(rows: i64, cols: i64, max_invocations: u32) -> Self {
        if max_invocations <= 1 || rows <= 0 || cols <= 0 {
            return Self::SERIAL;
        }
        let rows = rows as u32;
        let cols = cols as u32;
        let row_threads = max_invocations.min(rows);
        let col_threads = (max_invocations / row_threads).min(cols);
        Self {
            threads: [row_threads, col_threads],
            max_invocations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serial_asks_for_one_invocation() {
        assert_eq!(LaunchShape::SERIAL.count(), 1);
        assert!(!LaunchShape::SERIAL.is_parallel());
    }

    #[test]
    fn a_cap_below_two_stays_serial() {
        assert_eq!(LaunchShape::for_tile(32, 32, 1), LaunchShape::SERIAL);
        assert_eq!(LaunchShape::for_tile(32, 32, 0), LaunchShape::SERIAL);
    }

    #[test]
    fn rows_fill_before_columns() {
        let shape = LaunchShape::for_tile(4, 8, 4);
        assert_eq!(shape.threads, [4, 1]);
    }

    #[test]
    fn columns_only_take_what_the_rows_leave() {
        let shape = LaunchShape::for_tile(2, 8, 8);
        assert_eq!(shape.threads, [2, 4]);
        assert_eq!(shape.count(), 8);
    }

    #[test]
    fn a_small_tile_is_not_oversubscribed() {
        // One row of four columns can use four invocations, not eight.
        let shape = LaunchShape::for_tile(1, 4, 8);
        assert_eq!(shape.threads, [1, 4]);
        assert!(shape.count() <= 8);
    }

    #[test]
    fn the_product_never_exceeds_the_cap() {
        for rows in [1i64, 2, 3, 7, 32] {
            for cols in [1i64, 2, 5, 16] {
                for cap in [1u32, 2, 8, 64, 256] {
                    let shape = LaunchShape::for_tile(rows, cols, cap);
                    assert!(
                        shape.count() <= cap.max(1),
                        "{rows}x{cols} cap {cap} gave {shape:?}"
                    );
                }
            }
        }
    }
}
