# dealii-hermite

## Overview

I am interested in implementing Hermite elements in deal.II. This repository
contains a lot of in-progress code and experiments to this end.

## length-scale

One issue with Hermite elements is that the standard implementation has poor
conditioning: since, classically (see Ciarlet's book), the derivative basis
functions are chosen on each cell so that their derivatives are `O(1)`, they
tend to have function values that are `O(h)`: this results in a variety of
length scales in the matrix (and poor conditioning). This folder contains a
program that computes a local length scale for each vertex: this provides a
scaling factor for derivative basis functions (with derivative support points at
vertices) such that all basis functions have `O(1)` function values.
