#pragma once

#include "../neural_network/matrix.hpp"
#include <vector>
#include <memory>

class MemoryPool {
private:
    std::vector<std::unique_ptr<Matrix> > matrix_pool;
    std::vector<bool> available;
    size_t pool_size;
    
public:
    MemoryPool(size_t initial_size = 100);
    ~MemoryPool();
    
    // Get matrix from pool
    Matrix* get_matrix(size_t rows, size_t cols);
    
    // Return matrix to pool
    void return_matrix(Matrix* mat);
    
    // Pool management
    void resize_pool(size_t new_size);
    void clear_pool();
    
    // Statistics
    size_t get_pool_size() const { return pool_size; }
    size_t get_available_count() const;
    size_t get_used_count() const;
    
    // Utility
    void print_pool_status() const;
}; 