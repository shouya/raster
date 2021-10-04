use std::{collections::HashMap, rc::Rc};

use crate::{
  raster::Face,
  shader::Shader,
  types::{Mat4, Vec2, Vec3},
};

// A polygon vertex
#[derive(Clone, Default, PartialEq)]
pub struct PolyVert {
  pub pos: Vec3,
  pub uv: Option<Vec2>,
  pub normal: Option<Vec3>,
}

// I tricked a little to assume f32 can be compared for total equality
impl Eq for PolyVert {}

impl std::hash::Hash for PolyVert {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.pos.to_array().map(f32::to_bits).hash(state);
    self.uv.map(|x| x.to_array().map(f32::to_bits)).hash(state);
    self
      .normal
      .map(|x| x.to_array().map(f32::to_bits))
      .hash(state);
  }
}

impl From<(Vec3, Option<Vec2>, Option<Vec3>)> for PolyVert {
  fn from(v: (Vec3, Option<Vec2>, Option<Vec3>)) -> Self {
    Self {
      pos: v.0,
      uv: v.1,
      normal: v.2,
    }
  }
}

impl PolyVert {
  pub fn pos(pos: Vec3) -> Self {
    Self {
      pos,
      ..Default::default()
    }
  }
}

pub struct Mesh<T = PolyVert> {
  pub material: Option<Rc<dyn Shader>>,
  pub vertices: Vec<T>,
  pub faces: Vec<Face<usize>>,

  // index from vert to the index in the array
  index: Option<HashMap<T, usize>>,
}

impl<T> Clone for Mesh<T>
where
  T: Clone,
{
  fn clone(&self) -> Self {
    let Mesh {
      material,
      vertices,
      faces,
      index,
    } = self;

    let material = material
      .as_ref()
      .map(|x| Rc::from(dyn_clone::clone_box(&**x)));

    Mesh {
      material,
      index: index.clone(),
      vertices: vertices.clone(),
      faces: faces.clone(),
    }
  }
}

impl<T> Mesh<T> {
  #[allow(unused)]
  pub fn new() -> Self
  where
    T: Default,
  {
    Self {
      index: Some(HashMap::new()),
      material: Default::default(),
      vertices: Default::default(),
      faces: Default::default(),
    }
  }

  // A sealed mesh is allowed to perform "expensive"
  // operations like clone, as_ref, etc.
  //
  // However, a sealed mesh cannot add faces any more.
  pub fn is_sealed(&self) -> bool {
    self.index.is_none()
  }

  pub fn seal(&mut self) {
    self.index = None;
  }

  pub fn map_in_place<F>(&mut self, f: F)
  where
    F: Fn(&mut T),
  {
    self.vertices.iter_mut().for_each(f)
  }

  pub fn map<F, S>(self, f: F) -> Mesh<S>
  where
    F: Fn(T) -> S,
  {
    assert!(self.is_sealed());

    let vertices = self.vertices.into_iter().map(f).collect();
    Mesh {
      vertices,
      material: self.material,
      index: None,
      faces: self.faces,
    }
  }
}

impl<T> Mesh<T> {
  pub fn set_material(&mut self, material: impl Shader + 'static) {
    self.material = Some(Rc::new(material));
  }
}

impl Mesh<PolyVert> {
  #[allow(unused)]
  pub fn add_simple_face(&mut self, vertices: &[Vec3]) {
    assert!(!self.is_sealed());
    let mut face = Face::new(false);
    for i in 0..vertices.len() {
      let vert = PolyVert::pos(vertices[i]);
      face.add_vert(self.add_vert(vert));
    }
    self.faces.push(face);
  }

  pub fn add_face(&mut self, vertices: &[(Vec3, Option<Vec2>, Option<Vec3>)]) {
    assert!(!self.is_sealed());
    let mut face = Face::new(false);
    for i in 0..vertices.len() {
      let vert = PolyVert::from(vertices[i]);
      face.add_vert(self.add_vert(vert));
    }
    self.faces.push(face);
  }

  fn add_vert(&mut self, vert: PolyVert) -> usize {
    let index = self.index.as_mut().unwrap();

    match index.get(&vert) {
      Some(i) => *i,
      None => {
        let n = self.vertices.len();
        self.vertices.push(vert.clone());
        index.insert(vert, n);
        n
      }
    }
  }

  pub fn apply_transformation(&self, matrix: &Mat4) -> Self {
    assert!(self.is_sealed());

    let mut mesh = self.clone();
    mesh.map_in_place(|mut v| {
      v.pos = matrix.transform_point3(v.pos);
      if let Some(normal) = v.normal {
        v.normal = Some(matrix.transform_vector3(normal))
      }
    });
    mesh
  }
}
