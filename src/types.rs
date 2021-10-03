use std::cmp;

use glam;

use crate::util::f32_cmp;

pub type Vector4 = glam::Vec4;
pub type Vector3 = glam::Vec3A;
pub type Point3 = glam::Vec3A;
pub type Vector2 = glam::Vec2;

pub type Mat4 = glam::Mat4;

// a wrapper to make point3 Ord
#[derive(Debug, Clone, Copy)]
pub struct Point3Ord(Point3);

impl Point3Ord {
  pub fn new(pt: Point3) -> Self {
    Point3Ord(pt)
  }

  fn cmp_to(&self, other: &Self) -> cmp::Ordering {
    f32_cmp(&self.0.x, &other.0.x)
      .then(f32_cmp(&self.0.y, &other.0.y))
      .then(f32_cmp(&self.0.z, &other.0.z))
  }
}

impl PartialOrd for Point3Ord {
  fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
    Some(self.cmp_to(other))
  }
}

impl Ord for Point3Ord {
  fn cmp(&self, other: &Self) -> cmp::Ordering {
    self.cmp_to(other)
  }
}

impl PartialEq for Point3Ord {
  fn eq(&self, other: &Self) -> bool {
    self.cmp_to(other).is_eq()
  }
}

impl Eq for Point3Ord {}

impl std::hash::Hash for Point3Ord {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.0.x.to_bits().hash(state);
    self.0.y.to_bits().hash(state);
    self.0.z.to_bits().hash(state);
  }
}
