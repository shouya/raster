use std::cmp;

use glam;

use crate::util::f32_cmp;

pub use glam::{Vec4, Vec2, Vec3, Mat4, vec2, vec3, vec4};

// a wrapper to make point3 Ord
#[derive(Debug, Clone, Copy)]
pub struct Vec4Ord(Vec4);

impl Vec4Ord {
  pub fn new(pt: Vec4) -> Self {
    Vec4Ord(pt)
  }

  fn cmp_to(&self, other: &Self) -> cmp::Ordering {
    f32_cmp(&self.0.x, &other.0.x)
      .then(f32_cmp(&self.0.y, &other.0.y))
      .then(f32_cmp(&self.0.z, &other.0.z))
  }
}

impl PartialOrd for Vec4Ord {
  fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
    Some(self.cmp_to(other))
  }
}

impl Ord for Vec4Ord {
  fn cmp(&self, other: &Self) -> cmp::Ordering {
    self.cmp_to(other)
  }
}

impl PartialEq for Vec4Ord {
  fn eq(&self, other: &Self) -> bool {
    self.cmp_to(other).is_eq()
  }
}

impl Eq for Vec4Ord {}

impl std::hash::Hash for Vec4Ord {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.0.x.to_bits().hash(state);
    self.0.y.to_bits().hash(state);
    self.0.z.to_bits().hash(state);
  }
}
