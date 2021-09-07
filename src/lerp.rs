use nalgebra::Point3;

pub trait Lerp {
  fn lerp(&self, other: &Self, t: f32) -> Self;
}

impl Lerp for f32 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    self + (other - self) * t
  }
}

impl Lerp for i32 {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    self + ((other - self) as f32 * t).round() as Self
  }
}

impl Lerp for Point3<f32> {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    Point3::new(
      lerp(t, &self.x, &other.x),
      lerp(t, &self.y, &other.y),
      lerp(t, &self.z, &other.z),
    )
  }
}

pub struct Uniform<T> {
  pub value: T,
}

impl<T> Lerp for Uniform<T>
where
  T: Clone,
{
  fn lerp(&self, other: &Self, t: f32) -> Self {
    if t <= 0.5 {
      Self {
        value: self.value.clone(),
      }
    } else {
      Self {
        value: other.value.clone(),
      }
    }
  }
}

pub fn lerp<T>(t: f32, from: &T, to: &T) -> T
where
  T: Lerp,
{
  from.lerp(to, t)
}

struct LerpIter<'a, T> {
  from: &'a T,
  to: &'a T,
  curr: usize,
  max: usize,
  count: usize,
}

impl<'a, T> Iterator for LerpIter<'a, T>
where
  T: Lerp,
{
  type Item = T;
  fn next(&mut self) -> Option<T> {
    if self.curr > self.max {
      return None;
    }
    let t = self.curr as f32 / self.count as f32;
    let v = lerp(t, self.from, self.to);
    self.curr += 1;
    Some(v)
  }
}

// will not actually reach "to"
#[allow(unused)]
pub fn lerp_iter<'a, T>(
  from: &'a T,
  to: &'a T,
  count: usize,
) -> impl Iterator<Item = T> + 'a
where
  T: Lerp,
{
  LerpIter {
    from,
    to,
    count,
    max: count - 1,
    curr: 0,
  }
}

pub fn lerp_closed_iter<'a, T>(
  from: &'a T,
  to: &'a T,
  count: usize,
) -> impl Iterator<Item = T> + 'a
where
  T: Lerp,
{
  LerpIter {
    from,
    to,
    count,
    max: count,
    curr: 0,
  }
}
