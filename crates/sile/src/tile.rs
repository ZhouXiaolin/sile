#[derive(Clone, Copy, Debug)]
pub struct TileId(pub i64);

pub fn id() -> TileId {
    TileId(0)
}
