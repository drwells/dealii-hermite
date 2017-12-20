#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>


using namespace dealii;
bool has_common_vertex(const Point<2> &current_vertex,
                       const Triangulation<2>::cell_iterator cell)
{
  for (unsigned int vertex_n = 0; vertex_n < GeometryInfo<2>::vertices_per_cell; ++vertex_n)
    if (cell->vertex(vertex_n) == current_vertex)
      return true;
  return false;
}


int main()
{
  PolarManifold<2> polar_manifold;
  Triangulation<2> triangulation;
  GridGenerator::hyper_ball(triangulation);
  triangulation.set_all_manifold_ids_on_boundary(42);
  triangulation.set_manifold(42, polar_manifold);

  for (unsigned int iter_n = 0; iter_n < 5; ++iter_n)
    {
      for (auto cell : triangulation.active_cell_iterators())
        if (Utilities::generate_normal_random_number(0.5, 0.5) > 0.75)
          cell->set_refine_flag();
      triangulation.execute_coarsening_and_refinement();
    }

  std::ofstream out ("grid-1.eps");
  GridOut grid_out;
  grid_out.write_eps (triangulation, out);


  std::vector<double> neighbor_cell_diameters;
  std::vector<Triangulation<2>::cell_iterator> vertex_neighbors;
  std::vector<double> vertex_length_scales(triangulation.get_vertices().size());
  std::fill(vertex_length_scales.begin(), vertex_length_scales.end(),
            std::numeric_limits<double>::signaling_NaN());

  const Triangulation<2>::cell_iterator endc = triangulation.end();
  for (const Triangulation<2>::active_cell_iterator cell : triangulation.active_cell_iterators())
    {
      for (unsigned int vertex_n = 0; vertex_n < GeometryInfo<2>::vertices_per_cell; ++vertex_n)
        {
          vertex_neighbors.clear();
          vertex_neighbors.push_back(cell);

          const Point<2> &current_vertex = cell->vertex(vertex_n);
          if (!std::isnan(vertex_length_scales[cell->vertex_index(vertex_n)]))
            continue;

          // find all neighbors of the current point. This will usually be
          // four, but, for general unstructured grids, is unbounded.
          Triangulation<2>::cell_iterator walking_cell = cell;
          bool check_for_new_neighbor = true;
          while (check_for_new_neighbor)
            {
              check_for_new_neighbor = false;
              for (unsigned int neighbor_n = 0; neighbor_n < GeometryInfo<2>::faces_per_cell; ++neighbor_n)
                {
                  if (check_for_new_neighbor)
                    break;
                  const Triangulation<2>::cell_iterator neighbor = walking_cell->neighbor(neighbor_n);

                  if (neighbor == endc || (neighbor->active() && neighbor->is_artificial()))
                    continue;

                  if (neighbor->active())
                    {
                      if (has_common_vertex(current_vertex, neighbor) &&
                          (std::find(vertex_neighbors.begin(), vertex_neighbors.end(), neighbor)
                           == vertex_neighbors.end()))
                        {
                          vertex_neighbors.push_back(neighbor);
                          walking_cell = neighbor;
                          check_for_new_neighbor = true;
                        }
                    }
                  else
                    {
                      for (unsigned int child_n = 0; child_n < neighbor->n_children(); ++child_n)
                        {
                          const Triangulation<2>::cell_iterator child = neighbor->child(child_n);
                          if (child->active() &&
                              has_common_vertex(current_vertex, child) &&
                              (std::find(vertex_neighbors.begin(), vertex_neighbors.end(), child)
                               == vertex_neighbors.end()))
                            {
                              vertex_neighbors.push_back(child);
                              walking_cell = child;
                              check_for_new_neighbor = true;
                            }
                        }
                    }
                }
            }

          neighbor_cell_diameters.clear();
          for (const auto cell : vertex_neighbors)
            neighbor_cell_diameters.push_back(cell->diameter());
          std::sort(neighbor_cell_diameters.begin(), neighbor_cell_diameters.end());
          const double length_scale = std::accumulate(neighbor_cell_diameters.begin(),
                                                      neighbor_cell_diameters.end(), 0.0)
            /neighbor_cell_diameters.size();

          vertex_length_scales[cell->vertex_index(vertex_n)] = length_scale;

          std::cout << "Point: "
                    << current_vertex
                    << " number of adjacent cells: "
                    << vertex_neighbors.size()
                    << " length scale: "
                    << length_scale
                    << '\n';
        }
    }

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);

  MappingQGeneric<2> mapping(1);
  QTrapez<2> quadrature;
  FEValues<2> fe_values(mapping, fe, quadrature, update_values | update_quadrature_points);

  dof_handler.distribute_dofs(fe);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> cell_diameters(triangulation.n_active_cells());
  std::vector<types::global_dof_index> cell_dof_indices(fe.dofs_per_cell);
  unsigned int cell_n = 0;
  for (auto cell : dof_handler.active_cell_iterators())
    {
      cell_diameters[cell_n] = cell->diameter();
      ++cell_n;

      fe_values.reinit(cell);
      cell->get_dof_indices(cell_dof_indices);

      for (unsigned int dof_n = 0; dof_n < fe.dofs_per_cell; ++dof_n)
        {
          // part 1: figure out which quadrature point is associated with this DoF
          unsigned int support_point_n = numbers::invalid_unsigned_int;
          for (unsigned int q_point_n = 0; q_point_n < quadrature.size(); ++q_point_n)
            {
              if (fe_values.shape_value(dof_n, q_point_n) > 0.9)
                {
                  support_point_n = q_point_n;
                  break;
                }
            }
          AssertThrow(support_point_n != numbers::invalid_unsigned_int,
                      ExcInternalError());

          // part 2: figure out which vertex goes with this support point
          unsigned int global_vertex_n = numbers::invalid_unsigned_int;
          for (unsigned int vertex_n = 0; vertex_n < GeometryInfo<2>::vertices_per_cell;
               ++vertex_n)
            {
              if (cell->vertex(vertex_n).distance(fe_values.quadrature_point(support_point_n)) < 1.0e-10)
                {
                  global_vertex_n = cell->vertex_index(vertex_n);
                  break;
                }
            }

          AssertThrow(global_vertex_n != numbers::invalid_unsigned_int,
                      ExcInternalError());

          solution[cell_dof_indices[dof_n]] = vertex_length_scales[global_vertex_n];
        }
    }


      DataOut<2, DoFHandler<2>> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "length_scales");
      data_out.add_data_vector(cell_diameters, "cell_diameters");
      data_out.build_patches();

      std::ofstream output("grid.vtu");
      data_out.write_vtu(output);
}
