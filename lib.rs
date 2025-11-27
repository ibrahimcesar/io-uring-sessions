pub mod advanced;
pub mod categorical;

//! # Session-Typed io_uring Prototype
//!
//! This module demonstrates how session types can encode the ownership protocol
//! for completion-based I/O. The key insight is that buffer ownership transfer
//! is a protocol step that should be enforced at compile time.
//!
//! ## The Protocol
//!
//! ```text
//! User                    Kernel
//!  │                        │
//!  │── Submit(buffer) ─────►│  User loses access to buffer
//!  │                        │
//!  │   (buffer in flight)   │  ILLEGAL to touch buffer here!
//!  │                        │
//!  │◄── Complete(buffer) ───│  User regains access
//!  │                        │
//! ```
//!
//! ## Design Goals
//!
//! 1. **Buffer safety**: Cannot access a buffer while kernel owns it
//! 2. **Protocol adherence**: Must handle completions for all submissions  
//! 3. **Zero runtime overhead**: All checks at compile time
//! 4. **Scalable**: Handle multiple in-flight operations

use std::marker::PhantomData;

// ============================================================================
// PART 1: Basic Types
// ============================================================================

/// A buffer that we fully own. Can read/write freely.
#[derive(Debug)]
pub struct OwnedBuffer {
    data: Vec<u8>,
}

impl OwnedBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0; size],
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// A token representing a buffer that has been submitted to the kernel.
/// 
/// CRITICAL: This type has NO methods to access the buffer data!
/// The buffer is "gone" until we get it back via completion.
/// 
/// The PhantomData<*const u8> makes this !Send and !Sync by default,
/// which is appropriate since the buffer location is pinned.
#[derive(Debug)]
pub struct SubmittedBuffer {
    /// We keep the pointer so we can reconstitute the buffer on completion,
    /// but we expose NO way to dereference it.
    ptr: *mut u8,
    len: usize,
    capacity: usize,
    _marker: PhantomData<*const u8>,
}

// SubmittedBuffer intentionally has NO methods to access data

/// Result of a completed read operation
#[derive(Debug)]
pub struct ReadCompletion {
    pub bytes_read: usize,
}

/// Result of a completed write operation  
#[derive(Debug)]
pub struct WriteCompletion {
    pub bytes_written: usize,
}

// ============================================================================
// PART 2: Session Type Encoding via Typestate
// ============================================================================

/// Marker trait for protocol states
pub trait ProtocolState {}

/// State: Ready to submit an operation
pub struct Ready;
impl ProtocolState for Ready {}

/// State: Operation submitted, waiting for completion
/// The type parameter tracks WHAT we submitted (for type-safe completion)
pub struct Pending<Op> {
    _op: PhantomData<Op>,
}
impl<Op> ProtocolState for Pending<Op> {}

/// Operation types
pub struct ReadOp;
pub struct WriteOp;

/// A session-typed handle to an io_uring-like interface.
/// 
/// The state parameter S tracks where we are in the protocol.
/// This is a SINGLE-operation handle for simplicity.
pub struct IoSession<S: ProtocolState> {
    // In real implementation: io_uring handle
    _state: PhantomData<S>,
    // The submitted buffer, if any
    submitted: Option<SubmittedBuffer>,
}

impl IoSession<Ready> {
    /// Create a new session, ready to submit operations
    pub fn new() -> Self {
        Self {
            _state: PhantomData,
            submitted: None,
        }
    }

    /// Submit a read operation.
    /// 
    /// Notice the signature:
    /// - Takes `self` by value (consumes the Ready state)
    /// - Takes `buffer` by value (takes ownership!)
    /// - Returns `IoSession<Pending<ReadOp>>` (new state)
    /// 
    /// After this call:
    /// - The old IoSession<Ready> is gone (can't submit again)
    /// - The buffer is gone (can't access it!)
    /// - You MUST deal with the Pending state
    pub fn submit_read(self, buffer: OwnedBuffer, _fd: i32, _offset: u64) 
        -> IoSession<Pending<ReadOp>> 
    {
        // Convert owned buffer to submitted (loses access)
        let submitted = unsafe {
            let mut buf = std::mem::ManuallyDrop::new(buffer.data);
            SubmittedBuffer {
                ptr: buf.as_mut_ptr(),
                len: buf.len(),
                capacity: buf.capacity(),
                _marker: PhantomData,
            }
        };

        // In real impl: submit to io_uring SQ
        println!("  [kernel] Received read submission, buffer is now mine!");

        IoSession {
            _state: PhantomData,
            submitted: Some(submitted),
        }
    }

    /// Submit a write operation (similar pattern)
    pub fn submit_write(self, buffer: OwnedBuffer, _fd: i32, _offset: u64)
        -> IoSession<Pending<WriteOp>>
    {
        let submitted = unsafe {
            let mut buf = std::mem::ManuallyDrop::new(buffer.data);
            SubmittedBuffer {
                ptr: buf.as_mut_ptr(),
                len: buf.len(),
                capacity: buf.capacity(),
                _marker: PhantomData,
            }
        };

        println!("  [kernel] Received write submission, buffer is now mine!");

        IoSession {
            _state: PhantomData,
            submitted: Some(submitted),
        }
    }
}

impl IoSession<Pending<ReadOp>> {
    /// Wait for read completion and get the buffer back.
    /// 
    /// Returns:
    /// - The completion result
    /// - The buffer (ownership restored!)
    /// - A new Ready session
    pub fn await_completion(mut self) -> (ReadCompletion, OwnedBuffer, IoSession<Ready>) {
        // In real impl: wait on io_uring CQ
        println!("  [kernel] Read complete, returning buffer to user");

        let submitted = self.submitted.take().expect("buffer must exist");
        
        // Reconstitute the buffer - we get ownership back!
        let buffer = unsafe {
            OwnedBuffer {
                data: Vec::from_raw_parts(submitted.ptr, submitted.len, submitted.capacity),
            }
        };

        let completion = ReadCompletion { bytes_read: 42 }; // Simulated

        let new_session = IoSession {
            _state: PhantomData,
            submitted: None,
        };

        (completion, buffer, new_session)
    }
}

impl IoSession<Pending<WriteOp>> {
    /// Wait for write completion
    pub fn await_completion(mut self) -> (WriteCompletion, OwnedBuffer, IoSession<Ready>) {
        println!("  [kernel] Write complete, returning buffer to user");

        let submitted = self.submitted.take().expect("buffer must exist");
        
        let buffer = unsafe {
            OwnedBuffer {
                data: Vec::from_raw_parts(submitted.ptr, submitted.len, submitted.capacity),
            }
        };

        let completion = WriteCompletion { bytes_written: 42 };

        let new_session = IoSession {
            _state: PhantomData,
            submitted: None,
        };

        (completion, buffer, new_session)
    }
}

// ============================================================================
// PART 3: Multi-Operation Sessions (More Advanced)
// ============================================================================

/// Track multiple in-flight operations at the type level using type-level lists.
/// This is where it gets interesting!

/// Type-level natural numbers for indexing
pub struct Z;          // Zero
pub struct S<N>(PhantomData<N>);  // Successor

/// Type-level list of pending operations
pub trait PendingList {}

/// Empty list - no pending operations
pub struct PNil;
impl PendingList for PNil {}

/// Cons cell - one pending op followed by more
pub struct PCons<Op, Tail: PendingList> {
    _marker: PhantomData<(Op, Tail)>,
}
impl<Op, Tail: PendingList> PendingList for PCons<Op, Tail> {}

/// A session that can have MULTIPLE in-flight operations.
/// The type parameter tracks ALL pending operations.
pub struct MultiSession<Pending: PendingList> {
    _pending: PhantomData<Pending>,
    buffers: Vec<Option<SubmittedBuffer>>, // indexed storage
}

impl MultiSession<PNil> {
    pub fn new() -> Self {
        Self {
            _pending: PhantomData,
            buffers: Vec::new(),
        }
    }
}

impl<P: PendingList> MultiSession<P> {
    /// Submit a read, adding it to the pending list.
    /// Returns the index for later completion.
    pub fn submit_read(mut self, buffer: OwnedBuffer, _fd: i32, _offset: u64) 
        -> (usize, MultiSession<PCons<ReadOp, P>>) 
    {
        let idx = self.buffers.len();
        
        let submitted = unsafe {
            let mut buf = std::mem::ManuallyDrop::new(buffer.data);
            SubmittedBuffer {
                ptr: buf.as_mut_ptr(),
                len: buf.len(),
                capacity: buf.capacity(),
                _marker: PhantomData,
            }
        };
        
        self.buffers.push(Some(submitted));
        
        println!("  [kernel] Read #{} submitted", idx);

        (idx, MultiSession {
            _pending: PhantomData,
            buffers: self.buffers,
        })
    }
}

// For completion, we need to "remove" from the type-level list.
// This is where dependent types would really help!
// For now, we'll use a simpler approach with runtime index checking.

impl<Op, Tail: PendingList> MultiSession<PCons<Op, Tail>> {
    /// Complete the most recently submitted operation (LIFO for simplicity).
    /// A real implementation would use proper indexing.
    pub fn complete_last(mut self) -> (ReadCompletion, OwnedBuffer, MultiSession<Tail>) {
        let submitted = self.buffers.pop().unwrap().unwrap();
        
        let buffer = unsafe {
            OwnedBuffer {
                data: Vec::from_raw_parts(submitted.ptr, submitted.len, submitted.capacity),
            }
        };

        println!("  [kernel] Operation complete, buffer returned");

        (
            ReadCompletion { bytes_read: 42 },
            buffer,
            MultiSession {
                _pending: PhantomData,
                buffers: self.buffers,
            }
        )
    }
}

// ============================================================================
// PART 4: A Cleaner API Using a Builder Pattern
// ============================================================================

/// A more ergonomic API that still maintains safety
pub mod ergonomic {
    use super::*;

    /// Token proving we have a pending operation at index I
    pub struct PendingToken<const I: usize> {
        _private: (),
    }

    /// A handle that tracks pending ops at the type level using const generics
    pub struct Ring<const PENDING: usize> {
        buffers: Vec<Option<SubmittedBuffer>>,
    }

    impl Ring<0> {
        pub fn new() -> Self {
            Self { buffers: Vec::new() }
        }
    }

    impl<const N: usize> Ring<N> {
        /// Submit a read. The PENDING count increases by 1.
        pub fn submit_read(
            mut self, 
            buffer: OwnedBuffer, 
            _fd: i32, 
            _offset: u64
        ) -> (PendingToken<N>, Ring<{ N + 1 }>) {
            let submitted = unsafe {
                let mut buf = std::mem::ManuallyDrop::new(buffer.data);
                SubmittedBuffer {
                    ptr: buf.as_mut_ptr(),
                    len: buf.len(),
                    capacity: buf.capacity(),
                    _marker: PhantomData,
                }
            };
            
            self.buffers.push(Some(submitted));
            
            println!("  [kernel] Read submitted (now {} pending)", N + 1);

            (
                PendingToken { _private: () },
                Ring { buffers: self.buffers }
            )
        }
    }

    // Note: Completing with const generic index arithmetic is tricky.
    // In real code, you'd need more sophisticated type-level machinery.
    
    impl Ring<1> {
        pub fn complete(
            mut self,
            _token: PendingToken<0>,
        ) -> (ReadCompletion, OwnedBuffer, Ring<0>) {
            let submitted = self.buffers.pop().unwrap().unwrap();
            
            let buffer = unsafe {
                OwnedBuffer {
                    data: Vec::from_raw_parts(submitted.ptr, submitted.len, submitted.capacity),
                }
            };

            println!("  [kernel] Completed (now 0 pending)");

            (
                ReadCompletion { bytes_read: 42 },
                buffer,
                Ring { buffers: self.buffers }
            )
        }
    }
}

// ============================================================================
// TESTS / EXAMPLES
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_session() {
        println!("\n=== Basic Session Type Demo ===\n");
        
        // Create a buffer we own
        let mut buffer = OwnedBuffer::new(1024);
        buffer.as_mut_slice()[0] = 42;
        println!("1. Created buffer, wrote 42 to first byte");
        
        // Create a session
        let session: IoSession<Ready> = IoSession::new();
        println!("2. Created session in Ready state");
        
        // Submit a read - THIS CONSUMES BOTH session AND buffer
        let pending_session: IoSession<Pending<ReadOp>> = session.submit_read(buffer, 0, 0);
        println!("3. Submitted read - buffer is now with kernel");
        
        // COMPILE ERROR if uncommented:
        // buffer.as_slice();  // Error: buffer was moved!
        // session.submit_read(...);  // Error: session was moved!
        
        // Wait for completion - get buffer back!
        let (completion, returned_buffer, new_session) = pending_session.await_completion();
        println!("4. Got completion: {} bytes read", completion.bytes_read);
        println!("5. Buffer returned! First byte is still: {}", returned_buffer.as_slice()[0]);
        
        // Can submit again with new session
        let _ = new_session.submit_write(returned_buffer, 0, 0);
        println!("6. Submitted another operation with returned buffer");
    }

    #[test]
    fn test_multi_session() {
        println!("\n=== Multi-Operation Session Demo ===\n");
        
        let buf1 = OwnedBuffer::new(1024);
        let buf2 = OwnedBuffer::new(2048);
        
        let session = MultiSession::<PNil>::new();
        println!("1. Created multi-session");
        
        // Submit two operations
        let (idx1, session) = session.submit_read(buf1, 0, 0);
        let (idx2, session) = session.submit_read(buf2, 1, 0);
        println!("2. Submitted reads at indices {} and {}", idx1, idx2);
        
        // Type of session is now: MultiSession<PCons<ReadOp, PCons<ReadOp, PNil>>>
        // The type KNOWS we have two pending operations!
        
        // Complete in LIFO order (for this simple impl)
        let (comp2, buf2, session) = session.complete_last();
        println!("3. Completed second read: {} bytes", comp2.bytes_read);
        
        let (comp1, buf1, _session) = session.complete_last();
        println!("4. Completed first read: {} bytes", comp1.bytes_read);
        
        // Buffers are back!
        println!("5. Got buffers back: {} and {} bytes", 
            buf1.as_slice().len(), 
            buf2.as_slice().len()
        );
    }

    #[test] 
    fn test_ergonomic_api() {
        println!("\n=== Ergonomic API Demo ===\n");
        
        use ergonomic::*;
        
        let buffer = OwnedBuffer::new(1024);
        let ring: Ring<0> = Ring::new();
        
        // Submit - get token proving we have pending work
        let (token, ring): (PendingToken<0>, Ring<1>) = ring.submit_read(buffer, 0, 0);
        
        // Complete - must provide the token!
        let (completion, buffer, _ring): (_, _, Ring<0>) = ring.complete(token);
        
        println!("Completed with {} bytes, buffer size {}", 
            completion.bytes_read,
            buffer.as_slice().len()
        );
    }
}

// ============================================================================
// WHAT'S MISSING FOR PRODUCTION
// ============================================================================

/// This prototype demonstrates the core ideas but lacks:
/// 
/// 1. **Actual io_uring integration**: Replace simulation with real syscalls
/// 
/// 2. **Async/await support**: The completion methods should be async
///    ```ignore
///    async fn await_completion(self) -> (Completion, Buffer, Session<Ready>)
///    ```
/// 
/// 3. **Out-of-order completion**: Real io_uring completes in any order.
///    Need to track which index completed and return the right buffer.
///    This likely requires dependent types or runtime checking.
/// 
/// 4. **Registered buffers**: Pre-registered buffer pools are more complex.
///    The protocol becomes:
///    - Register pool → get pool handle
///    - Use buffer from pool → buffer "checked out"  
///    - Complete → buffer "checked in"
///    - Unregister pool → get all buffers back
/// 
/// 5. **Linked operations**: io_uring supports chains of dependent ops.
///    This maps beautifully to session type sequencing!
/// 
/// 6. **Multishot operations**: Some ops complete multiple times.
///    Need session types with loops/recursion.
/// 
/// 7. **Error handling**: Completions can fail. Need:
///    ```ignore
///    recv Completion {
///        Ok(result) => { ... },
///        Err(errno) => { ... buffer still returns ... }
///    }
///    ```
/// 
/// 8. **Linear type enforcement**: Rust is affine, not linear.
///    A leaked session means leaked buffers. Solutions:
///    - #[must_use] on all session types  
///    - Destructor that panics if pending
///    - True linear types (language change)
