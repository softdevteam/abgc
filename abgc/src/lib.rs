#![feature(alloc_layout_extra)]
#![feature(allocator_api)]
#![feature(coerce_unsized)]
#![feature(specialization)]
#![feature(unsize)]

use std::{
    alloc::{alloc, dealloc, Layout},
    mem::{align_of, forget, size_of},
    ops::Deref,
    ptr::{self, NonNull},
};

/// Since Rust's alloc system requires us to tell it the `Layout` of a region of memory upon
/// deallocation, we either have to store the `Layout` or calculate it when needed. We choose the
/// latter, which forces every GCable thing to implement this trait.
pub trait GcLayout {
    fn layout(&self) -> Layout;
}

#[derive(Debug)]
pub struct Gc<T: GcLayout> {
    objptr: NonNull<T>,
}

impl<T: GcLayout> Gc<T> {
    /// Create a `Gc` from `v`.
    pub fn new(v: T) -> Self {
        let objptr = Gc::<T>::alloc_blank(Layout::new::<T>());
        let gc = unsafe {
            objptr.as_ptr().copy_from_nonoverlapping(&v, 1);
            Gc::from_raw(objptr)
        };
        forget(v);
        gc
    }

    /// Allocate memory sufficient to `l` (i.e. correctly aligned and of at least the required
    /// size). The returned pointer must be passed to `Gc::from_raw`.
    pub fn alloc_blank(l: Layout) -> NonNull<T> {
        let (layout, uoff) = Layout::new::<usize>().extend(l).unwrap();
        // In order for our storage scheme to work, it's necessary that `uoff - sizeof::<usize>()`
        // gives a valid alignment for a `usize`. There are only two cases we need to consider
        // here:
        //   1) `object`'s alignment is smaller than or equal to `usize`. If so, no padding will be
        //      added, at which point by definition `uoff - sizeof::<usize>()` will be exactly
        //      equivalent to the start point of the layout.
        //   2) `object`'s alignment is bigger than `usize`. Since alignment must be a power of
        //      two, that means that we must by definition be adding at least one exact multiple of
        //      `usize` bytes of padding.
        // The assert below is thus paranoia writ large: it could only trigger if `Layout` started
        // adding amounts of padding that directly contradict the documentation.
        debug_assert_eq!(uoff % align_of::<usize>(), 0);

        unsafe {
            let baseptr = alloc(layout);
            let objptr = baseptr.add(uoff);
            let clonesptr = objptr.sub(size_of::<usize>());
            ptr::write(clonesptr as *mut usize, 1);
            NonNull::new_unchecked(objptr as *mut T)
        }
    }

    /// Consumes the `Gc` returning a pointer which can be later used to recreate a `Gc` using
    /// either `from_raw` or `clone_from_raw`. Failing to recreate the `Gc` will lead to a memory
    /// leak.
    pub fn into_raw(self) -> NonNull<T> {
        let objptr = self.objptr;
        forget(self);
        objptr
    }

    /// Create a `Gc` from a raw pointer previously created by `alloc_blank` or `into_raw`. Note
    /// that this does not increment the reference count.
    pub unsafe fn from_raw(objptr: NonNull<T>) -> Self {
        Gc { objptr }
    }

    /// Create a `Gc` from a raw pointer previously created by `into_raw`, incrementing the
    /// reference count at the same time.
    pub unsafe fn clone_from_raw(objptr: NonNull<T>) -> Self {
        let clonesptr = (objptr.as_ptr() as *mut u8).sub(size_of::<usize>()) as *mut usize;
        let clones = ptr::read(clonesptr);
        ptr::write(clonesptr, clones + 1);
        Gc { objptr }
    }

    /// Recreate the `Gc<T>` pointing to `valptr`. If `valptr` was not originally directly created
    /// by `Gc`/`GcBox` then undefined behaviour will result.
    pub unsafe fn recover(objptr: NonNull<T>) -> Self {
        Gc::clone_from_raw(objptr)
    }

    /// Clone the GC object `gcc`. Note that this is an associated method.
    pub fn clone(gcc: &Gc<T>) -> Self {
        unsafe {
            let clonesptr = (gcc.objptr.as_ptr() as *mut u8).sub(size_of::<usize>()) as *mut usize;
            let clones = ptr::read(clonesptr);
            ptr::write(clonesptr, clones + 1);
        }
        Gc { objptr: gcc.objptr }
    }

    /// Is `this` pointer equal to `other`?
    pub fn ptr_eq(this: &Gc<T>, other: &Gc<T>) -> bool {
        ptr::eq(this.deref(), other.deref())
    }

    #[cfg(test)]
    fn clones(&self) -> usize {
        unsafe {
            let clonesptr = (self.objptr.as_ptr() as *mut u8).sub(size_of::<usize>()) as *mut usize;
            ptr::read(clonesptr)
        }
    }
}

impl<T: GcLayout> Deref for Gc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.objptr.as_ref() }
    }
}

impl<T: GcLayout> Drop for Gc<T> {
    fn drop(&mut self) {
        let t_layout = self.layout();
        unsafe {
            let clonesptr = (self.objptr.as_ptr() as *mut u8).sub(size_of::<usize>()) as *mut usize;
            let clones = ptr::read(clonesptr);
            if clones == 1 {
                ptr::drop_in_place(self.objptr.as_ptr());
                let (layout, uoff) = Layout::new::<usize>().extend(t_layout).unwrap();
                let baseptr = (self.objptr.as_ptr() as *mut u8).sub(uoff);
                dealloc(baseptr, layout);
            } else {
                ptr::write(clonesptr, clones - 1)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! gclayout {
        ($(#[$attr:meta])* $n: ident) => {
            impl GcLayout for $n {
                fn layout(&self) -> std::alloc::Layout {
                    std::alloc::Layout::new::<$n>()
                }
            }
        };
    }

    gclayout!(i64);

    #[test]
    fn test_gc_new() {
        let v1 = Gc::new(42);
        assert_eq!(v1.clones(), 1);
        {
            let v2 = Gc::clone(&v1);
            assert_eq!(v1.clones(), 2);
            assert_eq!(v2.clones(), 2);
        }
        assert_eq!(v1.clones(), 1);
    }

    #[test]
    fn test_gc_nonnull() {
        assert_eq!(size_of::<Gc<i64>>(), size_of::<Option<Gc<i64>>>());
    }
}
