#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstddef>
#include <xmmintrin.h>

#include <stdexcept>


template <typename NumberType>
struct Alignment {
    constexpr static const std::size_t value =
        sizeof(NumberType) == sizeof(double) 
            ? 32
            : sizeof(NumberType) == sizeof(float)
                ? 16
                : 0;

    static_assert(value != 0, "Unknown alignment size");
};

template <typename T, std::size_t Alignment = Alignment<T>::value>
struct AlignedAllocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    AlignedAllocator() { }
    AlignedAllocator(const AlignedAllocator&) { }
    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) { }
    ~AlignedAllocator() { }

    AlignedAllocator& operator=(const AlignedAllocator&) = delete;

    pointer address(reference r) const {
        return &r;
    }

    const_pointer address(const_reference r) const {
        return &r;
    }

    template <typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    void construct(pointer p, const_reference r) const {
        new (p) value_type(r);
    }

    void destory(pointer const p) const {
        p->~value_type();
    }

    pointer allocate(const std::size_t n) const {
        if (!n) {
            return nullptr;
        }

        if (n > max_size()) {
            throw std::length_error("AlignedAllocator<T>::allocate() - Integer overflow.");
        }

        void* const p = _mm_malloc(n * sizeof(value_type), Alignment);

        if (!p) {
            throw std::bad_alloc();
        }

        return static_cast<pointer>(p);
    }

    void deallocate(pointer const p, const std::size_t) const {
        _mm_free(p);
    }

    bool operator==(const AlignedAllocator& other) const {
        return true;
    }

    bool operator!=(const AlignedAllocator& other) const {
        return !operator==(other);
    }

  private:
    std::size_t max_size() const {
        return (static_cast<std::size_t>(0) - static_cast<std::size_t>(1)) / sizeof(T);
    }
};
