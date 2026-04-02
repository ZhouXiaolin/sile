pub fn require_divisible(total: i64, tile: i64) -> crate::Result<()> {
    if total % tile == 0 {
        Ok(())
    } else {
        Err(crate::Error::Shape(format!(
            "shape {total} is not divisible by tile {tile}"
        )))
    }
}
