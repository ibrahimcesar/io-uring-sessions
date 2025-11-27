# io-uring-sessions

## 1. `lib.rs` — Basic Session Types

Shows the fundamental pattern:

```rust
// State is tracked in the type
let session: IoSession<Ready> = IoSession::new();

// Submit CONSUMES both session and buffer
let pending: IoSession<Pending<ReadOp>> = session.submit_read(buffer, fd, offset);

// Cannot access buffer here - it's been moved!
// buffer.as_slice();  // COMPILE ERROR

// Complete returns ownership
let (result, buffer, new_session) = pending.await_completion();
// Now buffer is back!
```

The key insight: **the type changes as the protocol progresses**, making illegal states unrepresentable.

## 2. `advanced.rs` — Type-Level Operation Tracking

Shows how to track multiple in-flight operations:

```rust
// Type-level natural numbers for operation IDs
pub struct Zero;
pub struct Succ<N>(PhantomData<N>);

// Type-level sets for tracking pending ops
pub struct Insert<E, S: NatSet>(PhantomData<(E, S)>);

// Each submission returns a receipt
let (receipt1, ring) = ring.submit_read(buf1, 0, 0);
let (receipt2, ring) = ring.submit_read(buf2, 1, 0);

// Can complete out of order, but must use correct receipt!
let (_, buf2, ring) = ring.complete(receipt2);  // OK
let (_, buf1, ring) = ring.complete(receipt1);  // OK
// ring.complete(receipt1);  // COMPILE ERROR - receipt was consumed
```

## 3. `categorical.rs` — The Mathematical Structure
Makes explicit the category theory:

| Concept | In Code | Categorically |
|--| --| --|
| States `IoState` trait| Objects|
| Operations| `Submit`, `Complete` | Morphisms| 
| Sequence| `;` |Composition (∘)| 
| Parallel| Independent operations| Tensor (⊗)| 
| Internal ID | Operation ID| Traced wire |

The **trace** structure is particularly elegant: the operation ID is "internal" — it's used to match submit with complete but doesn't appear in the final type signature.

# Next Steps to Make This Real

1. **Integrate with actual `io-uring` crate**: The simulation needs to become real syscalls
2. **Add async**: The `await_completion` should be async and integrate with Tokio/async-std
3. **Handle out-of-order completion properly**: Real io_uring completes in any order. The type system tracks this, but the runtime needs to match completions to the right receipts
4. **Registered buffer pools**: A more complex protocol where buffers are "checked out" from a pool
5. **Linked operations**: io_uring supports chains of dependent operations — this maps beautifully to session type sequencing
