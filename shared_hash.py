import multiprocessing
import numpy as np
import multiprocessing.shared_memory as shm

# Remove the fixed TABLE_SIZE and make it dynamic
INITIAL_SIZE = 1000
LOAD_FACTOR = 0.7  # When to resize (70% full)

# Create a lock for synchronizing access
lock = multiprocessing.Lock()

def init_shared_memory():
    """Initialize shared memory for the hash table.
    Creates a shared memory block of initial size * 8 bytes (64-bit floats).
    Initializes all entries to -1.0 to indicate empty slots.
    Returns a tuple of (shared memory object, current size).
    """
    shared_mem = shm.SharedMemory(create=True, size=INITIAL_SIZE * 8)
    array = np.ndarray((INITIAL_SIZE,), dtype=np.float64, buffer=shared_mem.buf)
    array.fill(-1.0)
    return shared_mem, INITIAL_SIZE

def resize_table(shared_mem_name, current_size):
    """Resize the shared memory hash table to double its current size.
    Args:
        shared_mem_name: Name of the current shared memory block
        current_size: Current size of the table
    Returns:
        Tuple of (new shared memory object, new size)
    """
    new_size = current_size * 2
    old_mem = shm.SharedMemory(name=shared_mem_name)
    old_array = np.ndarray((current_size,), dtype=np.float64, buffer=old_mem.buf)
    
    # Create new shared memory
    new_mem = shm.SharedMemory(create=True, size=new_size * 8)
    new_array = np.ndarray((new_size,), dtype=np.float64, buffer=new_mem.buf)
    new_array.fill(-1.0)
    
    # Rehash all existing elements
    with lock:
        for i in range(current_size):
            if old_array[i] != -1.0:
                # Rehash using new size
                index = hash(i) % new_size
                new_array[index] = old_array[i]
    
    # Cleanup old memory
    old_mem.close()
    old_mem.unlink()
    
    return new_mem, new_size

def insert(shared_mem_name, current_size, key, value):
    """Insert a key-value pair into the shared hash table.
    Automatically resizes the table if it becomes too full.
    Args:
        shared_mem_name: Name of the shared memory block
        current_size: Current size of the table
        key: The key to hash and store
        value: The floating-point value to store (must be > 0)
    Returns:
        Tuple of (new shared memory name, new size) if resized,
        or (shared_mem_name, current_size) if not resized
    """
    if value <= 0:
        raise ValueError("Value must be greater than 0")
        
    existing_mem = shm.SharedMemory(name=shared_mem_name)
    array = np.ndarray((current_size,), dtype=np.float64, buffer=existing_mem.buf)

    with lock:
        # Check load factor
        used_slots = np.count_nonzero(array != -1.0)
        if used_slots / current_size >= LOAD_FACTOR:
            existing_mem.close()
            return resize_table(shared_mem_name, current_size)
            
        index = hash(key) % current_size
        array[index] = value
        
    existing_mem.close()
    return shared_mem_name, current_size

def lookup(shared_mem_name, current_size, key):
    """Retrieve a value from the shared hash table.
    Args:
        shared_mem_name: Name of the shared memory block
        current_size: Current size of the table
        key: The key to look up
    Returns:
        The stored floating-point value or -1.0 if the slot is empty
    """
    existing_mem = shm.SharedMemory(name=shared_mem_name)
    array = np.ndarray((current_size,), dtype=np.float64, buffer=existing_mem.buf)
    
    with lock:
        index = hash(key) % current_size
        value = array[index]

    existing_mem.close()
    return value

def worker(shared_mem_name, current_size, key, value):
    """Worker function for inserting and retrieving values.
    Demonstrates concurrent access to the shared hash table.
    Args:
        shared_mem_name: Name of the shared memory block
        current_size: Current size of the table
        key: The key to insert/lookup
        value: The floating-point value to insert (must be > 0)
    """
    new_mem_name, new_size = insert(shared_mem_name, current_size, key, value)
    result = lookup(new_mem_name, new_size, key)
    print(f"Worker Process: Inserted {value}, Retrieved {result}")

if __name__ == "__main__":
    shared_mem, current_size = init_shared_memory()
    
    processes = []
    for i in range(5):  # Launch multiple processes
        p = multiprocessing.Process(target=worker, args=(shared_mem.name, current_size, i, i * 100))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Cleanup
    shared_mem.close()
    shared_mem.unlink()
