use std::{collections::HashMap, f32::INFINITY, rc::Rc, time::Instant};

use approx::abs_diff_eq;
use smallvec::{smallvec, SmallVec};

use crate::{
  lerp::{lerp, lerp_closed_iter, Lerp},
  mesh::{Mesh, PolyVert},
  shader::{
    Light, PureColor, Shader, ShaderContext, ShaderOptions, SimpleMaterial,
    TextureStash,
  },
  types::{Mat4, Vec2, Vec3, Vec4, Vec4Ord},
  util::{divw, divw3, f32_cmp},
};

pub type Color = Vec4;

#[allow(non_snake_case)]
pub(crate) mod COLOR {
  use super::*;

  pub fn rgb(r: f32, g: f32, b: f32) -> Color {
    rgba(r, g, b, 1.0)
  }

  pub fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
    glam::vec4(r, g, b, a)
  }
}

#[derive(Debug, Clone)]
pub struct Image<T> {
  dimension: (usize, usize),
  pixels: Vec<T>,
}

impl<T> Image<T> {
  pub fn new(size: (usize, usize)) -> Self
  where
    T: Default,
  {
    let len = size.0 * size.1;
    let mut buffer = Vec::with_capacity(len);
    for _ in 0..len {
      buffer.push(Default::default());
    }

    Self {
      dimension: size,
      pixels: buffer,
    }
  }

  pub fn new_filled(size: (usize, usize), val: T) -> Self
  where
    T: Clone,
  {
    let len = size.0 * size.1;
    let buffer = vec![val; len];

    Self {
      dimension: size,
      pixels: buffer,
    }
  }

  pub fn pixels(&self) -> impl Iterator<Item = &T> {
    self.pixels.iter()
  }

  pub fn width(&self) -> usize {
    self.dimension.0
  }

  pub fn height(&self) -> usize {
    self.dimension.1
  }

  pub fn pixels_with_coords(&self) -> impl Iterator<Item = ((i32, i32), &T)> {
    let w = self.width();
    self.pixels.iter().enumerate().map(move |(i, p)| {
      let x = i % w;
      let y = i / w;
      ((x as i32, y as i32), p)
    })
  }

  pub fn pixels_mut_with_coords(
    &mut self,
  ) -> impl Iterator<Item = ((i32, i32), &mut T)> {
    let w = self.width();
    self.pixels.iter_mut().enumerate().map(move |(i, p)| {
      let x = i % w;
      let y = i / w;
      ((x as i32, y as i32), p)
    })
  }

  pub fn pixels_mut(&mut self) -> impl Iterator<Item = &mut T> {
    self.pixels.iter_mut()
  }

  pub fn as_ref(&self) -> Image<&T> {
    Image {
      dimension: self.dimension,
      pixels: self.pixels.iter().collect(),
    }
  }

  pub fn pixel(&self, coords: (i32, i32)) -> Option<&T> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    self.pixels.get(idx as usize)
  }

  pub fn pixel_mut(&mut self, coords: (i32, i32)) -> Option<&mut T> {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return None;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    self.pixels.get_mut(idx as usize)
  }

  pub fn put_pixel(&mut self, coords: (i32, i32), value: T) {
    if coords.0 < 0
      || coords.1 < 0
      || coords.0 >= self.width() as i32
      || coords.1 >= self.height() as i32
    {
      return;
    }
    let idx = coords.1 * self.width() as i32 + coords.0;
    *self.pixels.get_mut(idx as usize).unwrap() = value;
  }

  pub fn map<F, S>(self, f: F) -> Image<S>
  where
    F: Fn(T) -> S,
  {
    let dimension = self.dimension;
    let pixels = self.pixels.into_iter().map(f).collect();
    Image { pixels, dimension }
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    self.pixels.iter_mut().for_each(f)
  }

  pub fn size(&self) -> (usize, usize) {
    self.dimension
  }

  pub fn size_f32(&self) -> (f32, f32) {
    (self.dimension.0 as f32, self.dimension.1 as f32)
  }
}

impl Image<Color> {
  pub fn nearest_color(&self, uv: Vec2) -> Color {
    let coords = self.nearest_coords(uv, (0, 0));
    *self.pixel(coords).unwrap()
  }

  pub fn bilinear_color(&self, uv: Vec2) -> Color {
    // row 1: |a b|
    // row 2: |c d|
    let color_a = *self.pixel(self.nearest_coords(uv, (0, 0))).unwrap();
    let color_b = *self.pixel(self.nearest_coords(uv, (1, 0))).unwrap();
    let color_c = *self.pixel(self.nearest_coords(uv, (0, 1))).unwrap();
    let color_d = *self.pixel(self.nearest_coords(uv, (1, 1))).unwrap();

    let x = uv.x * (self.width() - 1) as f32;
    let y = uv.y * (self.height() - 1) as f32;
    let xt = x - x.floor();
    let yt = y - y.floor();

    let color_l = lerp(yt, &color_a, &color_c);
    let color_r = lerp(yt, &color_b, &color_d);
    let color = lerp(xt, &color_l, &color_r);

    color
  }

  /// if out of bound, return the nearest coordinates within the bound
  fn nearest_coords(&self, uv: Vec2, offset: (i32, i32)) -> (i32, i32) {
    let mut x = (uv.x * (self.width() - 1) as f32) as i32 + offset.0;
    let mut y = (uv.y * (self.height() - 1) as f32) as i32 + offset.1;

    // if x < 0 {
    //   x = 0;
    // }
    // if y < 0 {
    //   y = 0;
    // }
    if x >= self.width() as i32 {
      x = self.width() as i32 - 1;
    }
    if y >= self.height() as i32 {
      y = self.height() as i32 - 1;
    }

    (x, y)
  }
}

#[derive(Debug, Clone)]
pub struct Camera {
  // world coordinate
  view: Mat4,
  projection: Mat4,
  matrix: Mat4,
}

impl Camera {
  pub fn new_perspective(
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
  ) -> Self {
    let projection = Mat4::perspective_rh_gl(fovy, aspect, znear, zfar);
    let view = Mat4::IDENTITY;
    let matrix = projection * view;
    Self {
      projection,
      view,
      matrix,
    }
  }

  pub fn view_matrix(&self) -> Mat4 {
    self.view
  }

  pub fn projection_matrix(&self) -> Mat4 {
    self.projection
  }

  pub fn transformd(&mut self, trans: &Mat4) {
    self.view *= trans.inverse();
    self.matrix = self.projection * self.view;
  }

  pub fn project(&self, pt: Vec4) -> Vec4 {
    self.matrix * pt
  }
}

#[derive(Debug, Clone, PartialEq, Hash, Eq, PartialOrd, Ord)]
pub struct Line<T> {
  ends: [T; 2],
}

impl<T> From<[T; 2]> for Line<T> {
  fn from(ends: [T; 2]) -> Self {
    Self { ends }
  }
}

impl<T> From<(T, T)> for Line<T> {
  fn from(ends: (T, T)) -> Self {
    Self {
      ends: [ends.0, ends.1],
    }
  }
}

impl<T> Line<T> {
  pub fn a(&self) -> &T {
    &self.ends[0]
  }
  pub fn b(&self) -> &T {
    &self.ends[1]
  }

  // reorder ends
  pub fn sort_ends(&mut self)
  where
    T: Ord,
  {
    self.ends.sort()
  }

  // The caller needs to ensure "self" is in screen coordinates
  pub fn to_horizontal_pixels(self) -> impl Iterator<Item = T>
  where
    T: GenericPoint,
  {
    // ensure the line is indeed flat
    debug_assert!((self.a().pos().y - self.b().pos().y) as i32 == 0);
    let x1 = self.a().pos().x;
    let x2 = self.b().pos().x;
    let w = (x1 - x2).abs() as usize;

    lerp_closed_iter(self.ends[0], self.ends[1], w + 1)
  }

  pub fn to_pixels(self) -> impl Iterator<Item = T>
  where
    T: GenericPoint,
  {
    let a = self.a().pos();
    let b = self.b().pos();
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let n = f32::max(dx.abs(), dy.abs()) as usize;
    lerp_closed_iter(self.ends[0], self.ends[1], n + 1)
  }

  pub fn clip(self) -> impl Iterator<Item = Line<T>>
  where
    T: GenericPoint,
  {
    self
      .clip_on_camera_plane()
      .into_iter()
      .flat_map(|mut line| {
        line.divw_in_place();
        line.clip_on_frustum()
      })
  }

  fn clip_on_camera_plane(mut self) -> impl Iterator<Item = Line<T>>
  where
    T: GenericPoint,
  {
    let get_w = |p: &Vec4| p.w;

    if !self.clip_component(get_w, 0.0001, INFINITY) {
      return None.into_iter();
    }

    Some(self).into_iter()
  }

  fn divw_in_place(&mut self)
  where
    T: GenericPoint,
  {
    self.map_in_place(|pt| *pt.pos_mut() = divw(*pt.pos()))
  }

  fn clip_on_frustum(mut self) -> impl Iterator<Item = Line<T>>
  where
    T: GenericPoint,
  {
    let get_x = |p: &Vec4| p.x;
    let get_y = |p: &Vec4| p.y;
    let get_z = |p: &Vec4| p.z;

    if !self.clip_component(get_x, -1.0, 1.0) {
      return None.into_iter();
    }
    if !self.clip_component(get_y, -1.0, 1.0) {
      return None.into_iter();
    }
    if !self.clip_component(get_z, -1.0, 1.0) {
      return None.into_iter();
    }

    Some(self).into_iter()
  }

  fn clip_component<F>(&mut self, get_comp: F, min: f32, max: f32) -> bool
  where
    T: GenericPoint,
    F: Fn(&Vec4) -> f32,
  {
    let mut av = get_comp(self.a().pos());
    let mut bv = get_comp(self.b().pos());

    if av < min && bv < min {
      // both beyond min; skip this line
      return false;
    }
    if av > max && bv > max {
      // both beyond max; skip this line
      return false;
    }
    if av >= min && av <= max && bv >= min && bv <= max {
      // within the range; draw without clipping
      return true;
    }

    if av < min && bv >= min {
      // clip a on min
      let t = (min - av) / (bv - av);
      debug_assert!((0.0..=1.0).contains(&t));
      self.ends[0] = lerp(t, self.a(), self.b());
    } else if av >= min && bv < min {
      // clip b on min
      let t = (min - av) / (bv - av);
      debug_assert!((0.0..=1.0).contains(&t));
      self.ends[1] = lerp(t, self.a(), self.b());
    }

    // recalculate because a and b may be changed
    av = get_comp(self.a().pos());
    bv = get_comp(self.b().pos());

    if av > max && bv <= max {
      // clip a on max
      let t = (max - av) / (bv - av);
      debug_assert!((0.0..=1.0).contains(&t));
      self.ends[0] = lerp(t, self.a(), self.b());
    } else if av <= max && bv > max {
      // clip b on max
      let t = (max - av) / (bv - av);
      debug_assert!((0.0..=1.0).contains(&t));
      self.ends[1] = lerp(t, self.a(), self.b());
    }

    true
  }

  pub fn map<S, F>(self, f: F) -> Line<S>
  where
    F: Fn(T) -> S,
  {
    Line {
      ends: self.ends.map(|x| f(x)),
    }
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    self.ends.iter_mut().for_each(f)
  }
}

#[derive(Debug, Clone)]
pub struct Trig<T> {
  vertices: [T; 3],
}

impl<T> From<[T; 3]> for Trig<T> {
  fn from(vertices: [T; 3]) -> Self {
    Self { vertices }
  }
}

impl<'a, T> Trig<T> {
  pub fn a(&self) -> &T {
    &self.vertices[0]
  }
  pub fn b(&self) -> &T {
    &self.vertices[1]
  }
  pub fn c(&self) -> &T {
    &self.vertices[2]
  }

  #[allow(unused)]
  pub fn edges(&'a self) -> impl Iterator<Item = Line<T>>
  where
    T: Clone,
  {
    vec![
      (self.vertices[0].clone(), self.vertices[1].clone()).into(),
      (self.vertices[1].clone(), self.vertices[2].clone()).into(),
      (self.vertices[2].clone(), self.vertices[0].clone()).into(),
    ]
    .into_iter()
  }

  pub fn vertices<S>(&'a self) -> [S; 3]
  where
    T: Clone,
    S: From<T>,
  {
    [
      From::from(self.vertices[0].clone()),
      From::from(self.vertices[1].clone()),
      From::from(self.vertices[2].clone()),
    ]
  }

  pub fn as_ref(&self) -> Trig<&T> {
    Trig {
      vertices: [self.a(), self.b(), self.c()],
    }
  }

  pub fn map<S, F>(self, f: F) -> Trig<S>
  where
    F: Fn(T) -> S,
  {
    Trig {
      vertices: self.vertices.map(|x| f(x)),
    }
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    self.vertices.iter_mut().for_each(f)
  }

  // the caller needs to ensure self.vert.pos are of screen coords
  pub fn to_fill_pixels(self) -> impl Iterator<Item = T>
  where
    T: GenericPoint,
  {
    let [upper, lower] = self.horizontally_split();
    let upper = upper
      .into_iter()
      .flat_map(|trig| trig.upper_trig_to_horizontal_lines());
    let lower = lower
      .into_iter()
      .flat_map(|trig| trig.lower_trig_to_horizontal_lines());
    let lines = upper.chain(lower);

    lines.flat_map(|line| line.to_horizontal_pixels())
  }

  pub fn to_edge_pixels(self) -> impl Iterator<Item = T>
  where
    T: GenericPoint,
  {
    self.edges().flat_map(|line| line.to_pixels())
  }

  // the caller needs to ensure self is an upper trig and
  // self.vert.pos are of screen coordinates
  pub fn upper_trig_to_horizontal_lines(self) -> impl Iterator<Item = Line<T>>
  where
    T: GenericPoint,
  {
    let top = self.a();
    let bottom_left = self.b();
    let bottom_right = self.c();

    let top_y = top.pos().y;
    let bottom_y = bottom_left.pos().y;
    let h = (top_y - bottom_y).abs() as usize;

    let left_pts_iter = lerp_closed_iter(*top, *bottom_left, h + 1);
    let right_pts_iter = lerp_closed_iter(*top, *bottom_right, h + 1);

    left_pts_iter.zip(right_pts_iter).map(Line::from)
  }

  pub fn lower_trig_to_horizontal_lines(self) -> impl Iterator<Item = Line<T>>
  where
    T: GenericPoint,
  {
    let top_left = self.a();
    let top_right = self.b();
    let bottom = self.c();

    let top_y = top_left.pos().y;
    let bottom_y = bottom.pos().y;
    let h = (top_y - bottom_y).abs() as usize;

    let left_pts_iter = lerp_closed_iter(*top_left, *bottom, h + 1);
    let right_pts_iter = lerp_closed_iter(*top_right, *bottom, h + 1);

    left_pts_iter.zip(right_pts_iter).map(Line::from)
  }

  // returns two triangles whose vertices are ordered by y values
  pub fn horizontally_split(mut self) -> [Option<Trig<T>>; 2]
  where
    T: GenericPoint,
  {
    const EPS: f32 = 0.1;
    self
      .vertices
      .sort_unstable_by(|p1, p2| f32_cmp(&p1.pos().y, &p2.pos().y));
    let ay = self.a().pos().y;
    let by = self.b().pos().y;
    let cy = self.c().pos().y;

    if abs_diff_eq!(ay, cy, epsilon = EPS) {
      // a flat line
      return [Some(self), None];
    }

    if abs_diff_eq!(ay, by, epsilon = EPS) {
      // a lower triangle
      return [None, Some(self)];
    }

    if abs_diff_eq!(by, cy, epsilon = EPS) {
      // a upper triangle
      return [Some(self), None];
    }

    // a normal triangle that we need to split
    let t = (by - ay) / (cy - ay);
    // dbg!(cy - ay);
    let ptl = lerp(t, self.a(), self.c());
    let ptr = *self.b();

    let upper_trig = [*self.a(), ptl, ptr].into();
    let lower_trig = [ptl, ptr, *self.c()].into();

    [Some(upper_trig), Some(lower_trig)]
  }

  pub fn clip(self) -> impl Iterator<Item = Self>
  where
    T: GenericPoint,
  {
    self
      .clip_on_camera_plane()
      .into_iter()
      .flat_map(|mut trig| {
        trig.divw_in_place();
        trig.clip_on_frustum()
      })
  }

  pub fn clip_on_frustum(self) -> SmallVec<[Trig<T>; 2]>
  where
    T: GenericPoint,
  {
    if self.fully_visible() {
      return smallvec![self];
    }

    let init: SmallVec<[&Trig<T>; 2]> = smallvec![&self];

    init
      .into_iter()
      .flat_map(|t| t.clip_component(|p| p.pos().z, -1.0, -1.0))
      .flat_map(|t| t.clip_component(|p| p.pos().z, 1.0, 1.0))
      .flat_map(|t| t.clip_component(|p| p.pos().x, -1.0, -1.0))
      .flat_map(|t| t.clip_component(|p| p.pos().x, 1.0, 1.0))
      .flat_map(|t| t.clip_component(|p| p.pos().y, -1.0, -1.0))
      .flat_map(|t| t.clip_component(|p| p.pos().y, 1.0, 1.0))
      .collect()
  }

  pub fn clip_on_camera_plane(self) -> impl Iterator<Item = Self>
  where
    T: GenericPoint,
  {
    // not clipping on 0.0 to avoid infinity
    self.clip_component(|p| p.pos().w, 0.001, -1.0).into_iter()
  }

  fn divw_in_place(&mut self)
  where
    T: GenericPoint,
  {
    self.map_in_place(|pt| *pt.pos_mut() = divw(*pt.pos()))
  }

  fn clip_component<F>(
    &self,
    get_comp: F,
    lim: f32,
    sign: f32,
  ) -> SmallVec<[Trig<T>; 2]>
  where
    T: GenericPoint,
    F: Fn(&T) -> f32,
  {
    let (va, vb, vc) = (self.a(), self.b(), self.c());
    let v: [f32; 3] = self.as_ref().map(get_comp).vertices();
    let (a, b, c) = (v[0], v[1], v[2]);

    // redefine lt, gt, le, ge operator based on the sign
    let lt = |x: f32, y: f32| x * sign < y * sign;
    let ge = |x: f32, y: f32| x * sign >= y * sign;
    let in_lim = |x: f32| lt(x, lim);
    let out_lim = |x: f32| ge(x, lim);

    // case 1: all vertex within range
    if in_lim(a) && in_lim(b) && in_lim(c) {
      return smallvec![self.clone()];
    }

    // case 2: all vertex out of range
    if out_lim(a) && out_lim(b) && out_lim(c) {
      // within the range; draw without clipping
      return smallvec![];
    }

    // case 3: two vertices out of range
    if in_lim(a) && out_lim(b) && out_lim(c) {
      let new_vb = lerp((lim - a) / (b - a), va, vb);
      let new_vc = lerp((lim - a) / (c - a), va, vc);
      return smallvec![[*va, new_vb, new_vc].into()];
    }
    if out_lim(a) && in_lim(b) && out_lim(c) {
      let new_va = lerp((lim - b) / (a - b), vb, va);
      let new_vc = lerp((lim - b) / (c - b), vb, vc);
      return smallvec![[new_va, *vb, new_vc].into()];
    }
    if out_lim(a) && out_lim(b) && in_lim(c) {
      let new_va = lerp((lim - c) / (a - c), vc, va);
      let new_vb = lerp((lim - c) / (b - c), vc, vb);
      return smallvec![[new_va, new_vb, *vc].into()];
    }

    // case 4: one vertex out of range
    if out_lim(a) && in_lim(b) && in_lim(c) {
      let new_vb = lerp((lim - a) / (b - a), va, vb);
      let new_vc = lerp((lim - a) / (c - a), va, vc);
      return smallvec![
        [*vb, *vc, new_vb].into(),
        [*vc, new_vc, new_vb].into()
      ];
    }
    if in_lim(a) && out_lim(b) && in_lim(c) {
      let new_va = lerp((lim - b) / (a - b), vb, va);
      let new_vc = lerp((lim - b) / (c - b), vb, vc);
      return smallvec![
        [*va, *vc, new_va].into(),
        [*vc, new_va, new_vc].into()
      ];
    }
    if in_lim(a) && in_lim(b) && out_lim(c) {
      let new_va = lerp((lim - c) / (a - c), vc, va);
      let new_vb = lerp((lim - c) / (b - c), vc, vb);
      return smallvec![
        [*va, *vb, new_va].into(),
        [*vb, new_va, new_vb].into()
      ];
    }

    unreachable!()
  }

  pub fn fully_visible(&self) -> bool
  where
    T: GenericPoint,
  {
    let comp_in_range = |v| (-1.0..=1.0).contains(&v);
    let pt_in_range = |p4: &Vec4| {
      let p3 = divw3(*p4);
      comp_in_range(p3.x) && comp_in_range(p3.y) && comp_in_range(p3.z)
    };

    pt_in_range(self.a().pos())
      && pt_in_range(self.b().pos())
      && pt_in_range(self.c().pos())
  }
}

pub trait GenericPoint: Lerp + Clone + Copy {
  fn pos(&self) -> &Vec4;
  fn pos_mut(&mut self) -> &mut Vec4;

  fn scale_to_screen(&mut self, (w, h): (f32, f32)) {
    let p = self.pos_mut();
    p.x = 0.5 * (w - 1.0) * (p.x + 1.0);
    p.y = 0.5 * (h - 1.0) * (1.0 - p.y);
  }
}

impl GenericPoint for Pt {
  fn pos(&self) -> &Vec4 {
    &self.pos
  }
  fn pos_mut(&mut self) -> &mut Vec4 {
    &mut self.pos
  }
}

impl GenericPoint for Vec4 {
  fn pos(&self) -> &Vec4 {
    &self
  }
  fn pos_mut(&mut self) -> &mut Vec4 {
    self
  }
}

#[derive(Debug, Clone)]
pub struct Face<T> {
  vertices: Vec<T>,
  double_faced: bool,
}

impl<T, const N: usize> From<[T; N]> for Face<T>
where
  T: Clone,
{
  fn from(verts: [T; N]) -> Self {
    Self {
      vertices: verts.into(),
      double_faced: false,
    }
  }
}

impl<T> From<&[T]> for Face<T>
where
  T: Clone,
{
  fn from(verts: &[T]) -> Self {
    Self {
      vertices: verts.into(),
      double_faced: false,
    }
  }
}

impl<T> From<Vec<T>> for Face<T> {
  fn from(vertices: Vec<T>) -> Self {
    Self {
      vertices,
      double_faced: false,
    }
  }
}

impl<T> Face<T> {
  pub fn new(double_faced: bool) -> Self {
    Face {
      vertices: Vec::new(),
      double_faced,
    }
  }
  pub fn add_vert(&mut self, v: T) {
    self.vertices.push(v);
  }

  pub fn vertices(&self) -> &[T] {
    self.vertices.as_ref()
  }

  pub fn double_faced(&self) -> bool {
    self.double_faced
  }

  pub fn set_double_faced(&mut self, double_faced: bool) {
    self.double_faced = double_faced;
  }

  /// the caller needs to ensure T is in clip space.
  pub fn is_hidden(&self) -> bool
  where
    T: GenericPoint,
  {
    if self.double_faced() {
      return false;
    }

    let positive_direction: Vec3 = [0.0, 0.0, 1.0].into();
    self.normal().dot(positive_direction) < 0.0
  }

  pub fn triangulate(&self) -> impl Iterator<Item = Trig<T>>
  where
    T: Copy,
  {
    debug_assert!(self.vertices.len() >= 3);

    let mut res = Vec::new();

    let v0 = self.vertices[0];
    let vs = self.vertices.iter().skip(1).collect::<Vec<_>>();
    for v in vs.windows(2) {
      res.push(Trig::from([v0, *v[0], *v[1]]));
    }

    res.into_iter()
  }

  pub fn edges(&self) -> impl Iterator<Item = Line<T>> + '_
  where
    T: Copy,
  {
    let n = self.vertices.len();
    (0..n).map(move |i| {
      let a = self.vertices[i];
      let b = self.vertices[(i + 1) % n];
      Line::from((a, b))
    })
  }

  pub fn normal(&self) -> Vec3
  where
    T: GenericPoint,
  {
    debug_assert!(self.vertices.len() >= 3);
    let p1 = divw3(*self.vertices()[0].pos());
    let p2 = divw3(*self.vertices()[1].pos());
    let p3 = divw3(*self.vertices()[2].pos());
    let v1 = p2 - p1;
    let v2 = p3 - p1;
    v1.cross(v2)
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T) -> (),
  {
    self.vertices.iter_mut().for_each(f)
  }

  #[allow(unused)]
  pub fn map<F, S>(self, f: F) -> Face<S>
  where
    F: Fn(T) -> S,
  {
    Face {
      vertices: self.vertices.into_iter().map(f).collect(),
      double_faced: self.double_faced,
    }
  }
}

impl Face<Pt> {
  pub fn world_normal(&self) -> Vec3 {
    debug_assert!(self.vertices.len() >= 3);
    let p1 = divw3(self.vertices()[0].world_pos);
    let p2 = divw3(self.vertices()[1].world_pos);
    let p3 = divw3(self.vertices()[2].world_pos);
    let v1 = p2 - p1;
    let v2 = p3 - p1;
    v1.cross(v2)
  }
}

#[derive(Clone)]
pub struct WorldMesh<T = PolyVert> {
  pub transform: Option<Mat4>,
  pub double_faced: bool,
  pub casts_shadow: bool,
  pub mesh: Rc<Mesh<T>>,
}

impl<T> From<Rc<Mesh<T>>> for WorldMesh<T> {
  fn from(mesh: Rc<Mesh<T>>) -> Self {
    assert!(mesh.is_sealed());

    Self {
      mesh,
      transform: None,
      double_faced: false,
      casts_shadow: true,
    }
  }
}

impl<T> From<Mesh<T>> for WorldMesh<T> {
  fn from(mesh: Mesh<T>) -> Self {
    assert!(mesh.is_sealed());

    Self {
      mesh: Rc::new(mesh),
      transform: None,
      double_faced: false,
      casts_shadow: true,
    }
  }
}

impl<T> WorldMesh<T> {
  pub fn double_faced(mut self, double_faced: bool) -> Self {
    self.double_faced = double_faced;
    self
  }

  pub fn casts_shadow(mut self, casts_shadow: bool) -> Self {
    self.casts_shadow = casts_shadow;
    self
  }

  pub fn faces(&self) -> impl Iterator<Item = Face<T>> + '_
  where
    T: Copy,
  {
    self.mesh.faces()
  }

  pub fn set_shader(mut self, shader: impl Shader + 'static) -> Self
  where
    T: Clone,
  {
    // notice: make_mut will clone a new instance of the shader
    Rc::make_mut(&mut self.mesh).set_material(shader);
    self
  }

  pub fn model_matrix(&self) -> Mat4 {
    self.transform.unwrap_or(Mat4::IDENTITY)
  }

  pub fn transformed(mut self, transform: Mat4) -> Self {
    let current_transform = self.transform.unwrap_or(Mat4::IDENTITY);
    self.transform = Some(transform * current_transform);
    self
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T),
    T: Clone,
  {
    Rc::make_mut(&mut self.mesh).map_in_place(f)
  }

  pub fn shader(&self) -> Rc<dyn Shader> {
    self
      .mesh
      .material
      .as_ref()
      .map(|x| x.clone())
      .unwrap_or_else(|| Rc::new(SimpleMaterial::plaster()))
  }
}

impl WorldMesh<PolyVert> {
  pub fn to_pt_mesh(&self) -> WorldMesh<Pt> {
    let Self {
      transform,
      double_faced,
      casts_shadow,
      ..
    } = self;
    let mesh = (&*self.mesh).clone();
    let mesh = Rc::new(mesh.map(|vert| Pt::from(vert)));

    WorldMesh {
      mesh,
      transform: transform.clone(),
      double_faced: double_faced.clone(),
      casts_shadow: casts_shadow.clone(),
    }
  }
}

pub struct Scene {
  textures: Rc<TextureStash>,
  camera: Camera,
  lights: Vec<Light>,
  meshes: Vec<WorldMesh<PolyVert>>,
}

impl Scene {
  pub fn new(camera: Camera) -> Self {
    Self {
      camera,
      textures: Rc::new(TextureStash::new()),
      lights: vec![],
      meshes: vec![],
    }
  }

  pub fn set_texture_stash(&mut self, textures: Rc<TextureStash>) {
    self.textures = textures;
  }

  pub fn texture_stash(&self) -> &TextureStash {
    &self.textures
  }

  pub fn add_light(&mut self, light: Light) {
    self.lights.push(light);
  }

  pub fn add_mesh(&mut self, mesh: WorldMesh) {
    self.meshes.push(mesh);
  }

  pub fn visualize_light(&mut self, mesh: Rc<Mesh>) {
    let lights = &self.lights;
    let meshes = &mut self.meshes;

    for light in lights {
      meshes.push(light.to_world_mesh(mesh.clone()).clone());
    }
  }

  pub fn iter_meshes(&self) -> impl Iterator<Item = &WorldMesh> + '_ {
    self.meshes.iter().map(|x| x)
  }

  pub fn camera(&self) -> &Camera {
    &self.camera
  }

  pub fn lights(&self) -> &[Light] {
    self.lights.as_slice()
  }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RasterizerMode {
  Wireframe,
  Shaded,
  Clipped,
}

impl Default for RasterizerMode {
  fn default() -> Self {
    RasterizerMode::Shaded
  }
}

/// A point on screen with integer xy coordinates and floating depth (z)
#[derive(PartialEq, Clone, Copy, Debug, Default)]
pub struct Pt {
  pub pos: Vec4,
  pub world_pos: Vec4,
  pub color: Color,
  pub normal: Vec3,
  pub uv: Vec2,
  pub in_shadow: Option<bool>,
  pub buf_v2: Option<Vec2>,
  pub buf_v3: Option<Vec3>,
}

impl From<PolyVert> for Pt {
  fn from(v: PolyVert) -> Self {
    let mut pt = Self::new(v.pos);
    if let Some(uv) = v.uv {
      pt.set_uv(uv);
    }
    if let Some(normal) = v.normal {
      pt.set_normal(normal);
    }
    pt
  }
}

impl Eq for Pt {}
impl std::hash::Hash for Pt {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.pos.to_array().map(f32::to_bits).hash(state);
    self.world_pos.to_array().map(f32::to_bits).hash(state);
    self.normal.to_array().map(f32::to_bits).hash(state);
    self.uv.to_array().map(f32::to_bits).hash(state);
    // other fields are omitted because these hashed properties should
    // be sufficient to determine a point
  }
}

impl From<Vec4> for Pt {
  fn from(v: Vec4) -> Self {
    Self {
      pos: v,
      world_pos: v,
      color: COLOR::rgba(1.0, 0.0, 0.0, 1.0),
      uv: Vec2::new(0.0, 0.0),
      normal: Vec3::ZERO,
      in_shadow: None,
      buf_v2: None,
      buf_v3: None,
    }
  }
}

impl Pt {
  pub fn new(point: Vec3) -> Self {
    Self {
      pos: (point, 1.0).into(),
      world_pos: (point, 1.0).into(),
      color: COLOR::rgba(1.0, 0.0, 0.0, 1.0),
      uv: Vec2::new(0.0, 0.0),
      normal: Vec3::ZERO,
      in_shadow: None,
      buf_v2: None,
      buf_v3: None,
    }
  }

  pub fn set_uv(&mut self, uv: Vec2) {
    self.uv = uv;
  }

  pub fn set_normal(&mut self, normal: Vec3) {
    self.normal = normal.normalize();
  }

  pub fn depth(&self) -> f32 {
    self.pos.z
  }
}

impl Lerp for Pt {
  fn lerp(&self, other: &Self, t: f32) -> Self {
    if self == other {
      return *self;
    }

    Pt {
      pos: lerp(t, &self.pos, &other.pos),
      world_pos: lerp(t, &self.world_pos, &other.world_pos),
      normal: lerp(t, &self.normal, &other.normal),
      color: lerp(t, &self.color, &other.color),
      uv: lerp(t, &self.uv, &other.uv),
      buf_v2: lerp(t, &self.buf_v2, &other.buf_v2),
      buf_v3: lerp(t, &self.buf_v3, &other.buf_v3),
      in_shadow: None,
    }
  }
}

#[derive(Debug, Default, Clone)]
pub struct RasterizerMetric {
  pub faces_rendered: usize,
  pub triangles_rendered: usize,
  pub clipped_triangles_rendered: usize,
  pub sub_triangles_rendered: usize,
  pub hidden_face_removed: usize,
  pub lines_drawn: usize,
  pub horizontal_lines_drawn: usize,
  pub vertices_shaded: usize,
  pub pixels_shaded: usize,
  pub pixels_discarded: usize,
  pub render_time: f32,
}

pub struct ShadowVolume {
  // the coords of volume are all in clip space
  volume: Mesh<Pt>,
  // key absent: new edge
  // >0: edge is light facing
  // <0: edge is light away
  // =0: edge is silhouette
  edge_index: HashMap<Line<Vec4Ord>, i32>,
  shadow_distance: f32,
  stencil: Image<i32>,
}

#[allow(unused)]
impl ShadowVolume {
  pub fn new(size: (usize, usize)) -> Self {
    Self {
      stencil: Image::new(size),
      volume: Mesh::new(),
      edge_index: HashMap::new(),
      shadow_distance: 10000.0,
    }
  }

  fn calc_face(
    &mut self,
    face: &Face<Pt>,
    camera: &Camera,
    light: &Light,
  ) -> Vec<Face<Vec4>> {
    let normal = face.world_normal();
    let sign = if normal.dot(*light.dir()) < 0.0 {
      -1
    } else {
      1
    };

    let mut res = Vec::new();

    for line in face.edges() {
      let key = self.register_edge(&line, sign);
      if !self.is_silhouette_edge(&key) {
        continue;
      }

      let p1 = line.a().world_pos;
      let p2 = line.b().world_pos;

      let p1_far = light.extrude(&p1, self.shadow_distance);
      let p2_far = light.extrude(&p2, self.shadow_distance);

      let face = [
        camera.project(p1),
        camera.project(p2),
        camera.project(p2_far),
        camera.project(p1_far),
      ];

      self.volume.add_face(&face);

      res.push(face.into());

      self.deregister_edge(&key);
    }

    res
  }

  pub fn add_world_mesh(
    &mut self,
    mesh: &WorldMesh<Pt>,
    depth: &Image<Option<f32>>,
    scene: &Scene,
  ) {
    assert!(scene.lights().len() == 1);
    let light = scene.lights().first().unwrap();
    let camera = scene.camera();

    if !mesh.casts_shadow {
      return;
    }

    for face in mesh.faces() {
      for shadow_face in self.calc_face(&face, camera, light) {
        self.rasterize_face(&shadow_face, depth);
      }
    }

    // different meshes don't share edges
    self.edge_index.clear();
  }

  fn rasterize_face(&mut self, face: &Face<Vec4>, zbuf: &Image<Option<f32>>) {
    let size = self.stencil.size_f32();
    let sign = if face.is_hidden() { 1 } else { -1 };

    for trig in face.triangulate() {
      for mut trig in trig.clip() {
        trig.map_in_place(|pt| pt.scale_to_screen(size));
        self.rasterize_trig(trig, zbuf, sign);
      }
    }
  }

  fn rasterize_trig(
    &mut self,
    trig: Trig<Vec4>,
    zbuf: &Image<Option<f32>>,
    sign: i32,
  ) {
    for pt in trig.to_fill_pixels() {
      let coords = (pt.x as i32, pt.y as i32);
      if let Some(depth) = *zbuf.pixel(coords).unwrap() {
        if pt.z < depth {
          let pixel = self.stencil.pixel_mut(coords).unwrap();
          *pixel += sign;
        }
      }
    }
  }

  fn register_edge(&mut self, line: &Line<Pt>, sign: i32) -> Line<Vec4Ord> {
    let mut key = line.clone().map(|x| Vec4Ord::new(x.world_pos));
    key.sort_ends();

    let val = self.edge_index.entry(key.clone()).or_insert(0);
    *val += sign;

    key
  }

  fn is_silhouette_edge(&self, key: &Line<Vec4Ord>) -> bool {
    let v = self.edge_index.get(key).unwrap_or(&0);
    *v == 0
  }

  fn deregister_edge(&mut self, key: &Line<Vec4Ord>) {
    self.edge_index.remove(key);
  }

  fn stencil(&self) -> &Image<i32> {
    &self.stencil
  }

  fn volume_mesh(&self) -> WorldMesh<Pt> {
    let mut volume = self.volume.clone();
    volume.seal();

    WorldMesh::from(volume)
      .set_shader(PureColor::new(COLOR::rgb(1.0, 0.0, 0.0)))
      .double_faced(true)
      .casts_shadow(false)
  }
}

#[derive(PartialEq, Clone, Copy)]
pub enum ShadowMode {
  NoShadow,
  RenderShadow,
  VisualizeShadowVolume,
}

pub struct Rasterizer {
  size: (f32, f32),
  mode: RasterizerMode,
  // smaller value = closer to camera; range: [-1, 1]
  zbuffer: Image<Option<f32>>,
  image: Image<Option<Pt>>,
  pixel_shader: Image<Option<Rc<dyn Shader>>>,
  metric: RasterizerMetric,
  shader_options: ShaderOptions,
  shadow_mode: ShadowMode,
  stencil_buffer: Option<Image<i32>>,
}

impl Rasterizer {
  pub fn new(size: (usize, usize)) -> Self {
    let image = Image::new_filled(size, None);
    let pixel_shader = Image::new_filled(size, None);
    let zbuffer = Image::new_filled(size, None);
    let mode = RasterizerMode::Shaded;
    let size = (image.width() as f32, image.height() as f32);
    let metric = Default::default();
    let shader_options = Default::default();

    Self {
      size,
      zbuffer,
      image,
      pixel_shader,
      mode,
      metric,
      shader_options,
      stencil_buffer: None,
      shadow_mode: ShadowMode::NoShadow,
    }
  }

  pub fn set_shader_options(&mut self, options: ShaderOptions) {
    self.shader_options = options;
  }

  pub fn set_mode(&mut self, mode: RasterizerMode) {
    self.mode = mode;
  }

  pub fn set_shadow_mode(&mut self, shadow_mode: ShadowMode) {
    self.shadow_mode = shadow_mode;
  }

  fn meshes_with_vertex_shaded(&self, scene: &Scene) -> Vec<WorldMesh<Pt>> {
    let mut context = self.shader_context(scene);

    scene
      .iter_meshes()
      .map(|world_mesh| {
        context.update_model_matrix(world_mesh.model_matrix());
        let shader = world_mesh.shader();

        let mut mesh_in_pt = world_mesh.to_pt_mesh();
        mesh_in_pt.map_in_place(|mut pt| shader.vertex(&context, &mut pt));
        mesh_in_pt
      })
      .collect()
  }

  pub fn rasterize(&mut self, scene: &Scene) {
    let now = Instant::now();

    let meshes = self.meshes_with_vertex_shaded(scene);

    match self.shadow_mode {
      ShadowMode::NoShadow => self.rasterize_pixels(&meshes),
      ShadowMode::RenderShadow => {
        let mut shadow = ShadowVolume::new(self.size_usize());
        self.rasterize_pixels(&meshes);
        self.rasterize_shadow(scene, &meshes, &mut shadow);
        self.save_shadow_info_in_pixels(shadow.stencil());
        self.stencil_buffer = Some(shadow.stencil().clone());
      }
      ShadowMode::VisualizeShadowVolume => {
        let mut shadow = ShadowVolume::new(self.size_usize());
        self.rasterize_pixels(&meshes);
        self.rasterize_shadow(scene, &meshes, &mut shadow);
        self.save_shadow_info_in_pixels(shadow.stencil());
        self.stencil_buffer = Some(shadow.stencil().clone());
        self.rasterize_shadow_volume(&shadow.volume_mesh());
      }
    }

    self.shade_pixels(scene);

    self.metric.render_time = now.elapsed().as_secs_f32();
  }

  #[inline(never)]
  pub fn rasterize_shadow(
    &mut self,
    scene: &Scene,
    meshes: &[WorldMesh<Pt>],
    shadow_volume: &mut ShadowVolume,
  ) {
    let depth = &self.zbuffer;
    for mesh in meshes.iter() {
      shadow_volume.add_world_mesh(&mesh, depth, scene);
    }
  }

  fn save_shadow_info_in_pixels(&mut self, stencil_buffer: &Image<i32>) {
    for (coords, pixel) in self.image.pixels_mut_with_coords() {
      if let Some(pt) = pixel {
        let stencil_value = stencil_buffer.pixel(coords).unwrap();
        pt.in_shadow = Some(*stencil_value != 0);
      }
    }
  }

  // Pt.pos field should be in clip space.
  pub fn rasterize_pixels(&mut self, meshes: &[WorldMesh<Pt>]) {
    for mesh in meshes.iter() {
      let shader = mesh.shader();

      for face in mesh.faces() {
        if face.is_hidden() {
          self.metric.hidden_face_removed += 1;
          continue;
        }

        self.metric.faces_rendered += 1;
        self.rasterize_face(&face, shader.clone());
      }
    }
  }

  fn rasterize_shadow_volume(&mut self, mesh: &WorldMesh<Pt>) {
    let shader = mesh.shader();

    for mut face in mesh.faces() {
      // make sure shadow volume is drawn above the actual mesh
      face.map_in_place(|p| p.pos.z -= 0.000001);

      // draw wireframe
      for line in face.edges() {
        self.draw_line(line, &shader);
      }
    }
  }

  pub fn rasterize_face(&mut self, face: &Face<Pt>, shader: Rc<dyn Shader>) {
    use RasterizerMode::*;

    match self.mode {
      Shaded => {
        for trig in face.triangulate() {
          self.fill_triangle(trig, &shader);
        }
      }
      Clipped => {
        for trig in face.triangulate() {
          self.draw_triangle_clipped(trig, &shader);
        }
      }
      Wireframe => {
        for line in face.edges() {
          self.draw_line(line, &shader);
        }
      }
    }
  }

  fn shade_pixels(&mut self, scene: &Scene) {
    let context = self.shader_context(scene);
    let pixels = self.image.pixels_mut().zip(self.pixel_shader.pixels());
    for pixel in pixels {
      match pixel {
        (None, None) => (),
        (Some(pt), Some(shader)) => {
          Self::shade_pixel(pt, &context, shader.as_ref())
        }
        _ => unreachable!(),
      }
    }
  }

  pub fn zbuffer_image(&self) -> Image<Color> {
    let to_comp = |d: Option<f32>| (d.unwrap_or(2.0) + 1.0) / 2.0;
    let to_color = |d| COLOR::rgb(to_comp(d), to_comp(d), to_comp(d));
    self.zbuffer.clone().map(to_color)
  }

  pub fn stencil_buffer_image(&self) -> Image<Color> {
    let size = self.size_usize();
    let default_image = Image::new_filled(size, 0);
    let image = self.stencil_buffer.as_ref().unwrap_or(&default_image);
    let to_color = |v: &i32| {
      let c = if *v < 0 {
        1.0
      } else if *v > 0 {
        0.0
      } else {
        0.5
      };
      COLOR::rgb(c, c, c)
    };
    image.as_ref().map(to_color)
  }

  pub fn metric(&self) -> RasterizerMetric {
    self.metric.clone()
  }

  pub fn into_image(self) -> Image<Color> {
    let to_color =
      |p: Option<Pt>| p.map(|x| x.color).unwrap_or(COLOR::rgb(0.0, 0.0, 0.0));
    self.image.map(to_color)
  }

  pub fn size_f32(&self) -> (f32, f32) {
    self.size
  }

  fn shade_pixel(pt: &mut Pt, context: &ShaderContext, shader: &dyn Shader) {
    // TODO: make metric modifiable without mutability
    // self.metric.pixels_shaded += 1;
    shader.fragment(context, pt)
  }

  // only put pixel in image buffer. does not shade it with color yet
  fn rasterize_pixel(&mut self, pt: Pt, shader: &Rc<dyn Shader>) {
    let (x, y) = (pt.pos.x as i32, pt.pos.y as i32);
    let (w, h) = self.size();

    if x < 0 || x >= w || y < 0 || y >= h {
      return;
    }

    let coords = (x, y);
    match self.zbuffer.pixel(coords).unwrap() {
      Some(d) if pt.depth() >= *d => {
        self.metric.pixels_discarded += 1;
        return;
      }
      None | Some(_) => {
        self.metric.pixels_shaded += 1;

        self.image.put_pixel(coords, Some(pt));
        self.pixel_shader.put_pixel(coords, Some(shader.clone()));

        self.zbuffer.put_pixel(coords, Some(pt.depth()));
      }
    }
  }

  fn draw_line(&mut self, line: Line<Pt>, shader: &Rc<dyn Shader>) {
    for mut line in line.clip() {
      line.map_in_place(|pt| pt.scale_to_screen(self.size_f32()));

      for pt in line.to_pixels() {
        self.rasterize_pixel(pt, shader);
      }
    }
  }

  // fill a triangle that is flat at bottom
  fn shader_context<'b>(&self, scene: &'b Scene) -> ShaderContext<'b> {
    let projection_mat4 = scene.camera().projection_matrix();
    let view_mat4 = scene.camera().view_matrix();
    let model_mat4 = Mat4::IDENTITY;
    let clip_mat4 = projection_mat4 * view_mat4 * model_mat4;

    ShaderContext {
      view_mat4,
      projection_mat4,
      model_mat4,
      clip_mat4,
      textures: scene.texture_stash(),
      lights: scene.lights(),
      options: self.shader_options.clone(),
    }
  }

  fn draw_triangle_clipped(&mut self, trig: Trig<Pt>, shader: &Rc<dyn Shader>) {
    for mut trig in trig.clip() {
      trig.map_in_place(|pt| pt.scale_to_screen(self.size_f32()));

      for pt in trig.to_edge_pixels() {
        self.rasterize_pixel(pt, shader);
      }
    }
  }

  fn fill_triangle(&mut self, trig: Trig<Pt>, shader: &Rc<dyn Shader>) {
    self.metric.triangles_rendered += 1;

    for mut trig in trig.clip() {
      trig.map_in_place(|pt| pt.scale_to_screen(self.size_f32()));

      for pt in trig.to_fill_pixels() {
        self.rasterize_pixel(pt, shader);
      }
    }
  }

  fn size(&self) -> (i32, i32) {
    (self.size.0 as i32, self.size.1 as i32)
  }

  fn size_usize(&self) -> (usize, usize) {
    (self.size.0 as usize, self.size.1 as usize)
  }
}

#[cfg(test)]
mod test {
  use std::f32::consts::PI;

  use super::*;
  use glam::{f32::Mat4, vec4};

  #[test]
  fn test_camera() {
    let fov = 120.0_f32.to_radians();
    let perspective = Mat4::perspective_rh_gl(fov, 1.0, 1.0, 100000000.0);

    dbg!(perspective * vec4(0.0, 0.0, 2.0, 1.0));
    dbg!(perspective * vec4(0.0, 0.0, 1.0, 1.0));
    dbg!(perspective * vec4(0.0, 0.0, 0.0, 1.0));
    dbg!(perspective * vec4(0.0, 0.0, -1.0, 1.0));
    dbg!(perspective * vec4(0.0, 0.0, -2.0, 1.0));
    dbg!(perspective * vec4(0.0, 0.0, -3.0, 1.0));

    assert!(false);
  }
}
