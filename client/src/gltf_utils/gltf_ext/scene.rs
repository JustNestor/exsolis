//! Scene graph helpers from glTF.

use std::collections::{HashMap, HashSet};
use fixedbitset::FixedBitSet;
use gltf::scene::Node;
use crate::gltf_utils::convert_coordinates::{convert_gltf_matrix, convert_gltf_transform};
use glam::{Mat4, Quat, Vec3};

/// Checks all glTF nodes for cycles, starting at the scene root.
pub fn check_for_cycles(document: &gltf::Document) -> Result<(), String> {
    let mut roots = FixedBitSet::with_capacity(document.nodes().len());
    for root in document.scenes().flat_map(|scene| scene.nodes()) {
        roots.insert(root.index());
    }

    let mut visited = FixedBitSet::with_capacity(document.nodes().len());
    for root in roots.ones() {
        let Some(node) = document.nodes().nth(root) else {
            continue;
        };
        check_is_part_of_cycle(&node, &mut visited)?;
    }

    Ok(())
}

/// Check if a node is part of a cycle
fn check_is_part_of_cycle(node: &Node, visited: &mut FixedBitSet) -> Result<(), String> {
    if visited.contains(node.index()) {
        return Err(format!("glTF nodes form a cycle involving node {}", node.index()));
    }
    visited.insert(node.index());
    for kid in node.children() {
        check_is_part_of_cycle(&kid, visited)?;
    }
    visited.remove(node.index());
    Ok(())
}

/// Calculates the world transform of a glTF node hierarchy.
pub fn compute_world_transforms(document: &gltf::Document) -> Vec<Mat4> {
    let nodes: Vec<Node> = document.nodes().collect();
    let mut world = vec![Mat4::IDENTITY; nodes.len()];
    
    let parent = parent_map(&nodes);
    
    // Find scene roots
    let scene = document.default_scene().expect("GLTF has no default scene");
    for root in scene.nodes().map(|n| n.index()) {
        compute_world_recursive(&nodes, root, Mat4::IDENTITY, &mut world);
    }
    
    world
}

/// Recursive world transform computation
fn compute_world_recursive(nodes: &[Node], index: usize, parent_world: Mat4, world: &mut [Mat4]) {
    let node = &nodes[index];
    let local = node_transform(node);
    let node_world = parent_world * local;
    world[index] = node_world;
    
    for child in node.children() {
        compute_world_recursive(nodes, child.index(), node_world, world);
    }
}

/// Parent map for node hierarchy
fn parent_map(nodes: &[Node]) -> Vec<Option<usize>> {
    let mut parent = vec![None; nodes.len()];
    for node in nodes {
        for child in node.children() {
            parent[child.index()] = Some(node.index());
        }
    }
    parent
}

/// Returns the local transform of a glTF node.
pub fn node_transform(node: &Node) -> Mat4 {
    match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => {
            convert_gltf_matrix(&matrix)
        }
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => {
            convert_gltf_transform(&translation, &rotation, &scale)
        }
    }
}

/// Collects the path from root to each node (for animation target resolution).
pub fn collect_node_paths(document: &gltf::Document) -> HashMap<usize, Vec<String>> {
    let mut paths = HashMap::new();
    let mut visited = HashSet::new();
    
    let scene = document.default_scene().expect("GLTF has no default scene");
    for root in scene.nodes() {
        collect_path_recursive(&root, &mut Vec::new(), &mut paths, &mut visited);
    }
    
    paths
}

fn collect_path_recursive(
    node: &Node,
    current_path: &mut Vec<String>,
    paths: &mut HashMap<usize, Vec<String>>,
    visited: &mut HashSet<usize>,
) {
    let name = node.name().map(ToString::to_string).unwrap_or_else(|| format!("Node{}", node.index()));
    current_path.push(name.clone());
    visited.insert(node.index());
    
    paths.insert(node.index(), current_path.clone());
    
    for child in node.children() {
        if !visited.contains(&child.index()) {
            collect_path_recursive(&child, current_path, paths, visited);
        }
    }
    
    current_path.pop();
    visited.remove(&node.index());
}

/// Skin data extracted from glTF
#[derive(Debug, Clone)]
pub struct SkinData {
    pub joints: Vec<usize>,           // Node indices of joints
    pub inverse_bind_matrices: Vec<Mat4>, // Inverse bind matrices
}

/// Extracts skin data from a glTF skin
pub fn extract_skin_data(
    skin: &gltf::Skin,
    buffers: &[gltf::buffer::Data],
) -> Result<SkinData, String> {
    let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
    
    let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));
    let inverse_bind_matrices: Vec<Mat4> = reader
        .read_inverse_bind_matrices()
        .ok_or("Skin missing inverse bind matrices")?
        .map(|m| convert_gltf_matrix(&m))
        .collect();
    
    if inverse_bind_matrices.len() != joints.len() {
        return Err("Inverse bind matrices count doesn't match joints count".into());
    }
    
    Ok(SkinData {
        joints,
        inverse_bind_matrices,
    })
}