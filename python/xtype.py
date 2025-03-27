"""
XType - A Python implementation of the xtype binary format

This module provides functionality for serializing Python data structures to the xtype binary format,
a universal binary notation language designed for efficient data exchange and storage.
"""

import struct
import numpy as np
from typing import Any, Dict, List, Tuple, Union, BinaryIO, Iterator, Optional

# xtype grammar
#--------------
#
# <file>       ::= <EOF> | <object> <EOF>
# <object>     ::= <content> | <footnote> <content>
# <footnote>   ::= "*" <content> | "*" <content> <footnote>
# <content>    ::= <element> | <list> | <dict>
# <list>       ::= "[]" | "[" <list_items> "]" | "[" <EOF> | "[" <list_items> <EOF>
# <list_items> ::= <object> | <object> <list_items>
# <dict>       ::= "{}" | "{" <dict_items> "}" | "{" <EOF> | "{" <list_items> <EOF>
# <dict_items> ::= <element> <object> | <element> <object> <dict_items>
# <element>    ::= <type> <bin_data> | "T"  | "F" | "n"
# <type>       ::= <lenght> <type> | <bin_data>
# <lenght>     ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" |
#                  "M" <bin_data> | "N" <bin_data> | "O" <bin_data> | "P" <bin_data>
# <bin_type>   ::= "i" | "j" | "k" | "l" | "I" | "J" | "K" | "L" |
#                  "b" | "h" | "f" | "d" | "s" | "u" | "S" | "x"

# <bin_data> is the binary data of defined size, according to the list of types.
# <EOF> is the end of file. In streams this could also be defined by a zero byte.


class XTypeFile:
    """
    A class for reading and writing Python data structures to files using the xtype binary format.

    Supports serialization of:
    - Basic types: int, float, str, bytes, bool
    - Container types: list, dict
    - NumPy arrays: 1D, 2D, and higher-dimensional arrays with various data types
    """

    def __init__(self, filename: str, mode: str = 'w'):
        """
        Initialize an XTypeFile object.

        Args:
            filename: Path to the file
            mode: File mode ('w' for write, 'r' for read)
        """
        self.filename = filename
        self.mode = mode
        self.file = None

        # Type mapping between Python/NumPy types and xtype format types
        self.type_map = {
            # Python -> xtype type mappings
            bool: 'b',
            int: self._select_int_type,  # Function to select appropriate int type
            float: 'd',
            str: 's',
            bytes: 'x',

            # NumPy -> xtype type mappings
            np.dtype('bool'): 'b',
            np.dtype('int8'): 'i',
            np.dtype('int16'): 'j',
            np.dtype('int32'): 'k',
            np.dtype('int64'): 'l',
            np.dtype('uint8'): 'I',
            np.dtype('uint16'): 'J',
            np.dtype('uint32'): 'K',
            np.dtype('uint64'): 'L',
            np.dtype('float16'): 'h',
            np.dtype('float32'): 'f',
            np.dtype('float64'): 'd',
        }

        # Size in bytes for each type
        self.type_sizes = {
            'i': 1, 'j': 2, 'k': 4, 'l': 8,  # signed ints
            'I': 1, 'J': 2, 'K': 4, 'L': 8,  # unsigned ints
            'b': 1,  # boolean
            'h': 2, 'f': 4, 'd': 8,  # floating point
            's': 1,  # string (utf-8)
            'u': 2,  # utf-16
            'S': 1,  # struct type as array of bytes
            'x': 1,  # other byte array
        }

        # Grammar terminal symbols (single character markers)
        self.grammar_terminals = set('*[]{}TFNijklIJKLbhfdsuSxMNOP0123456789')

    def __enter__(self):
        """Context manager entry point."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()

    def open(self):
        """Open the file for reading or writing."""
        if self.mode == 'w':
            self.file = open(self.filename, 'wb')
        elif self.mode == 'r':
            self.file = open(self.filename, 'rb')
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def close(self):
        """Close the file."""
        if self.file and not self.file.closed:
            self.file.close()

    def write(self, data: Any):
        """
        Write a Python object to the file in xtype format.

        Args:
            data: The Python object to serialize (can be a primitive type,
                 list, dict, or numpy array)
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for writing")

        self._write_object(data)

    def read_raw(self, byteorder: str = 'little') -> Iterator[Tuple[str, int, int]]:
        """
        Iterator to read type information from an xtype file without consuming binary data.

        This method parses the file according to the xtype grammar and yields a tuple with:
        1. A string representing the symbol or type
        2. An integer flag indicating:
           - 0: No length or size information
           - 1: Length information
           - 2: Data size information
        3. The length or data size (0 if there's no length or size)

        If binary data is associated with the yielded type, the caller must call
        read_raw_data() to consume the data or the subsequent read_raw() call
        will skip over it automatically.

        Args:
            byteorder: The byte order of multi-byte integers in the file. Defaults to 'little'.

        Yields:
            Tuple[str, int, int]: (symbol/type, flag, length_or_size)
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        # State tracking for binary data
        self._pending_binary_size = 0

        # Track accumulated length multipliers for arrays
        length_multiplier = 1

        while True:
            # Skip any pending binary data from previous call if not consumed
            if self._pending_binary_size > 0:
                self.file.seek(self._pending_binary_size, 1)  # Seek relative to current position
                self._pending_binary_size = 0

            # Read one byte
            char_byte = self.file.read(1)

            # Check for EOF
            if not char_byte:
                break

            try:
                char = char_byte.decode('ascii')
            except UnicodeDecodeError:
                # If we can't decode as ASCII, it's likely binary data that wasn't properly skipped
                # This can happen with string arrays where the binary data contains non-ASCII characters
                raise ValueError(f"Encountered non-ASCII character in grammar. This may indicate binary data wasn't properly skipped.")

            # Handle grammar terminal symbols
            if char in '[]{}TFn*':
                yield (char, 0, 0)
                continue

            # Handle direct length information (0-9)
            if char in '0123456789':
                yield (char, 1, int(char))
                # Multiply this length multiplier
                length_multiplier *= int(char)
                continue

            # Handle length information (M, N, O, P)
            if char in 'MNOP':
                size = {'M': 1, 'N': 2, 'O': 4, 'P': 8}[char]
                # binary_position = self.file.tell()
                binary_data = self.file.read(size)

                if len(binary_data) < size:
                    raise ValueError(f"Unexpected end of file when reading length of type {char}")

                # Convert binary to integer value based on type
                if char == 'M':  # uint8
                    value = binary_data[0]
                elif char == 'N':  # uint16
                    value = int.from_bytes(binary_data, byteorder=byteorder, signed=False)
                elif char == 'O':  # uint32
                    value = int.from_bytes(binary_data, byteorder=byteorder, signed=False)
                elif char == 'P':  # uint64
                    value = int.from_bytes(binary_data, byteorder=byteorder, signed=False)

                # Set pending binary size to 0 since we already consumed the binary data
                self._pending_binary_size = 0

                # Yield the length information and size
                yield (char, 1, value)

                # Multiply to length multiplier
                length_multiplier *= value
                continue

            # Handle data types
            if char in self.type_sizes:
                # For actual data types, calculate the total data size
                type_size = self.type_sizes[char]

                # Calculate total size based on accumulated length multiplier
                total_size = type_size * length_multiplier

                # Don't read the binary data yet, just note its size
                self._pending_binary_size = int(total_size)
                self._pending_binary_type = char

                yield (char, 2, total_size)
                length_multiplier = 1  # Reset length multiplier after using it
                continue

            # If we get here, we encountered an unexpected character
            raise ValueError(f"Unexpected character in xtype file: {repr(char)}")

    def read_raw_data(self, max_bytes: int = None) -> bytes:
        """
        Read the binary data that corresponds to the last type yielded by read_raw().

        This method must be called after read_raw() yields a type with associated
        binary data. If not called, the next read_raw() call will skip over the
        binary data automatically.

        Args:
            max_bytes: Maximum number of bytes to read. If None or greater than
                      remaining bytes, all remaining bytes are read. If less than
                      the total pending bytes, subsequent calls to read_raw_data
                      can read the remaining bytes, or they will be skipped by
                      the next read_raw() call.

        Returns:
            bytes: The binary data corresponding to the last type (up to max_bytes)

        Raises:
            ValueError: If there is no pending binary data to read
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        if self._pending_binary_size <= 0:
            raise ValueError("No pending binary data to read. Call read_raw() first.")

        # Determine how many bytes to read
        bytes_to_read = self._pending_binary_size
        if max_bytes is not None and max_bytes < bytes_to_read:
            bytes_to_read = max_bytes

        # Read the binary data
        binary_data = self.file.read(bytes_to_read)
        if len(binary_data) < bytes_to_read:
            raise ValueError(f"Unexpected end of file when reading data of type {self._pending_binary_type}")

        # Update the pending binary size
        self._pending_binary_size -= bytes_to_read

        return binary_data

    def _write_object(self, obj: Any):
        """
        Write an object to the file.

        Args:
            obj: The object to write
        """
        if isinstance(obj, (list, tuple)):
            self._write_list(obj)
        elif isinstance(obj, dict):
            self._write_dict(obj)
        elif isinstance(obj, np.ndarray):
            self._write_numpy_array(obj)
        elif isinstance(obj, bytes):
            # Handle bytes directly instead of treating as an element
            # This is important for binary data handling
            self._write_element(obj)
        elif obj is None:
            # Handle None explicitly
            self.file.write(b'n')
        else:
            self._write_element(obj)

    def _write_list(self, lst: List):
        """
        Write a list to the file.

        Args:
            lst: The list to write
        """
        self.file.write(b'[')
        for item in lst:
            self._write_object(item)
        self.file.write(b']')

    def _write_dict(self, d: Dict):
        """
        Write a dictionary to the file.

        Args:
            d: The dictionary to write
        """
        self.file.write(b'{')
        for key, value in d.items():
            # Convert key to string if it's not already
            if not isinstance(key, str):
                key = str(key)
            # Write the key as a string element
            self._write_element(key)
            # Write the value
            self._write_object(value)
        self.file.write(b'}')

    def _write_element(self, value: Any):
        """
        Write a basic element to the file.

        Args:
            value: The value to write
        """
        if value is None:
            self.file.write(b'n')
        elif isinstance(value, bool):
            self.file.write(b'b')
            self.file.write(b'\xff' if value else b'\x00')
        elif isinstance(value, int):
            type_code = self._select_int_type(value)
            self.file.write(type_code.encode())
            self._write_int_value(value, type_code)
        elif isinstance(value, float):
            self.file.write(b'd')
            self.file.write(struct.pack('>d', value))
        elif isinstance(value, str):
            # Write string with length prefix
            encoded = value.encode('utf-8')
            self._write_length(len(encoded))
            self.file.write(b's')
            self.file.write(encoded)
        elif isinstance(value, bytes):
            # Write bytes with length prefix
            self._write_length(len(value))
            self.file.write(b'x')
            self.file.write(value)
        else:
            raise TypeError(f"Unsupported type: {type(value)}")

    def _write_numpy_array(self, arr: np.ndarray):
        """
        Write a NumPy array to the file.

        Args:
            arr: The NumPy array to write
        """
        # Handle array dimensions
        for dim in arr.shape:
            self._write_length(dim)

        # Get the type code for the array's data type
        dtype = arr.dtype

        # Special handling for string arrays
        if np.issubdtype(dtype, np.string_) or np.issubdtype(dtype, np.unicode_):
            # For string arrays, we need to also write the string length dimension
            # Extract the itemsize which represents the max string length
            str_length = dtype.itemsize
            self._write_length(str_length)

            # For string arrays, use 's' type code
            self.file.write(b's')

            # Ensure the array is in C-contiguous order for efficient serialization
            if not arr.flags.c_contiguous:
                arr = np.ascontiguousarray(arr)

            # Write the entire array memory to the file
            self.file.write(arr.tobytes())

            return

        if dtype not in self.type_map:
            raise TypeError(f"Unsupported NumPy dtype: {dtype}")

        type_code = self.type_map[dtype]
        self.file.write(type_code.encode())

        # Ensure the array is in C-contiguous order for efficient serialization
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        # Write the array data based on its type
        if dtype == np.dtype('bool'):
            # Convert boolean array to bytes (0x00 for False, 0xFF for True)
            self.file.write(np.where(arr, 0xFF, 0x00).astype(np.uint8).tobytes())
        elif np.issubdtype(dtype, np.integer):
            # Handle integer types
            if type_code in ('i', 'I'):  # uint8, int8
                self.file.write(arr.tobytes())
            elif type_code in ('j', 'J'):  # uint16, int16
                self.file.write(arr.astype(dtype).byteswap('>').tobytes())
            elif type_code in ('k', 'K'):  # uint32, int32
                self.file.write(arr.astype(dtype).byteswap('>').tobytes())
            elif type_code in ('l', 'L'):  # uint64, int64
                self.file.write(arr.astype(dtype).byteswap('>').tobytes())
        elif np.issubdtype(dtype, np.floating):
            # Handle floating point types
            if type_code == 'h':  # float16
                self.file.write(arr.astype(np.float16).byteswap('>').tobytes())
            elif type_code == 'f':  # float32
                self.file.write(arr.astype(np.float32).byteswap('>').tobytes())
            elif type_code == 'd':  # float64
                self.file.write(arr.astype(np.float64).byteswap('>').tobytes())

    def _select_int_type(self, value: int) -> str:
        """
        Select the appropriate integer type code based on the value.

        Args:
            value: The integer value

        Returns:
            The xtype type code
        """
        if value >= 0:
            if value <= 0xFF:
                return 'I'  # uint8
            elif value <= 0xFFFF:
                return 'J'  # uint16
            elif value <= 0xFFFFFFFF:
                return 'K'  # uint32
            else:
                return 'L'  # uint64
        else:
            if value >= -0x80 and value <= 0x7F:
                return 'i'  # int8
            elif value >= -0x8000 and value <= 0x7FFF:
                return 'j'  # int16
            elif value >= -0x80000000 and value <= 0x7FFFFFFF:
                return 'k'  # int32
            else:
                return 'l'  # int64

    def _write_int_value(self, value: int, type_code: str):
        """
        Write an integer value with the specified type code.

        Args:
            value: The integer value
            type_code: The xtype type code
        """
        if type_code == 'I':
            self.file.write(struct.pack('>B', value))
        elif type_code == 'J':
            self.file.write(struct.pack('>H', value))
        elif type_code == 'K':
            self.file.write(struct.pack('>I', value))
        elif type_code == 'L':
            self.file.write(struct.pack('>Q', value))
        elif type_code == 'i':
            self.file.write(struct.pack('>b', value))
        elif type_code == 'j':
            self.file.write(struct.pack('>h', value))
        elif type_code == 'k':
            self.file.write(struct.pack('>i', value))
        elif type_code == 'l':
            self.file.write(struct.pack('>q', value))

    def _write_length(self, length: int):
        """
        Write a length value using the appropriate format.

        Args:
            length: The length to write
        """
        if length <= 9:
            # Single-digit lengths are written as ASCII characters '0' through '9'
            self.file.write(str(length).encode())
        elif length <= 0xFF:
            # uint8 length
            self.file.write(b'M')
            self.file.write(struct.pack('>B', length))
        elif length <= 0xFFFF:
            # uint16 length
            self.file.write(b'N')
            self.file.write(struct.pack('>H', length))
        elif length <= 0xFFFFFFFF:
            # uint32 length
            self.file.write(b'O')
            self.file.write(struct.pack('>I', length))
        else:
            # uint64 length
            self.file.write(b'P')
            self.file.write(struct.pack('>Q', length))

    def read_debug(self, indent_size: int = 2, max_indent_level: int = 10, byteorder: str = 'little', max_binary_bytes: int = 15) -> Iterator[str]:
        """
        Iterator to read raw data from an xtype file and convert each output to a formatted string.

        This method uses read_raw and read_raw_data to parse the xtype file and formats
        the output as a string where:
        - The string part is enclosed in quotation marks
        - If the string part ends with 's', the binary data is converted to a UTF-8 string with quotation marks
          unless it's part of a multidimensional array
        - Otherwise, the bytes are converted to hexadecimal with spaces in between
        - Indentation is added based on brackets [ and ] and curly brackets { and }
        - Special bracket characters are always on their own line
        - String parts with empty binary data are gathered with a single pair of quotation marks
        - Length values are included in parentheses when present

        Args:
            indent_size: Number of spaces per indentation level (default: 2)
            max_indent_level: Maximum indentation level (default: 10)
            byteorder: Byte order for integer values (default: 'little')
            max_binary_bytes: Maximum number of binary bytes to read (default: 10)

        Yields:
            str: A formatted string for each element in the xtype file
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        # Reset the file position at the start
        self.file.seek(0)

        # Initialize internal state
        indent_level = 0
        # Collecting characters that don't have binary data
        accumulated_strings = []
        # Track array dimensions to detect multidimensional arrays
        dimensions = []
        # Flag to indicate if we're inside an array context
        in_array_context = False

        try:
            for symbol, flag, length_or_size in self.read_raw(byteorder=byteorder):
                # Handle special bracket characters
                if symbol in '[]{}':
                    # First, if we have accumulated strings, yield them with the current indentation
                    if accumulated_strings:
                        # Join all accumulated strings without spaces and wrap in a single pair of quotes
                        accumulated_str = "".join(accumulated_strings)
                        yield ' ' * min(indent_level, max_indent_level) * indent_size + f'{accumulated_str}'
                        accumulated_strings = []

                    # For closing brackets, decrease indentation before printing
                    if symbol in ']}':
                        indent_level = max(0, indent_level - 1)
                        # If exiting an array context, reset dimensions
                        if symbol == ']' and in_array_context:
                            # Only reset if we're closing the outermost array
                            if indent_level == 0:
                                dimensions = []
                                in_array_context = False

                    # Print the bracket on its own line with proper indentation
                    yield ' ' * min(indent_level, max_indent_level) * indent_size + f'{symbol}'

                    # For opening brackets, increase indentation after printing
                    if symbol in '[{':
                        indent_level += 1
                        # Mark that we're entering an array context if it's a square bracket
                        if symbol == '[':
                            in_array_context = True

                    continue

                # For non-bracket characters with no binary data, accumulate them
                if flag == 0:
                    accumulated_strings.append(symbol)
                    continue
                elif flag == 1:
                    # If it's a length value (0-9 or M,N,O,P with value), include it in parentheses
                    accumulated_strings.append(f"{symbol}({length_or_size})" if symbol in "MNOP" else f"{symbol}")
                    # If we're in array context and get a length value, it could be a dimension
                    if in_array_context and indent_level > 0:
                        dimensions.append(length_or_size)
                    continue

                # If we get here, it's a data type with binary data (flag == 2)
                current_indent = ' ' * min(indent_level, max_indent_level) * indent_size

                # Include accumulated strings if any
                if accumulated_strings:
                    # Join all accumulated strings without spaces and add the current symbol
                    accumulated_str = "".join(accumulated_strings) + symbol
                    accumulated_strings = []
                else:
                    accumulated_str = symbol

                # Format based on accumulated_str
                if accumulated_str.endswith('s'):
                    # Get the string data and format them
                    binary_part = self.read_raw_data(max_bytes=max_binary_bytes) if length_or_size > 0 else b''
                    # Check if we're in a multidimensional array context
                    is_multidimensional = in_array_context and len(dimensions) > 1

                    # For string type, convert binary to UTF-8 string with quotation marks
                    # only if not part of a multidimensional array
                    try:
                        string_value = binary_part.decode('utf-8', errors='replace')
                        if is_multidimensional:
                            # Treat like any other non-string array - show as hex
                            hex_str = ' '.join(f'{b:02x}' for b in binary_part)
                            if len(binary_part) < length_or_size:
                                hex_str += f" ... ({length_or_size} bytes total)"
                            yield current_indent + f'{accumulated_str}: {hex_str}'
                        else:
                            # Regular string display with quotation marks
                            yield current_indent + f'{accumulated_str}: "{string_value}"'
                    except Exception:
                        # If decoding fails, fall back to hex representation
                        hex_str = ' '.join(f'{b:02x}' for b in binary_part)
                        if len(binary_part) < length_or_size:
                            hex_str += f" ... ({length_or_size} bytes total)"
                        yield current_indent + f'{accumulated_str}: {hex_str}'
                else:
                    # Get the data (limited by max_binary_bytes) and format them
                    binary_part = self.read_raw_data(max_bytes=max_binary_bytes) if length_or_size > 0 else b''
                    # For other types, convert to space-separated hex
                    hex_str = ' '.join(f'{b:02x}' for b in binary_part)
                    if len(binary_part) < length_or_size:
                        hex_str += f" ... ({length_or_size} bytes total)"
                    yield current_indent + f'{accumulated_str}: {hex_str}'
        except Exception as e:
            # If we have any accumulated strings when an exception occurs, output them
            if accumulated_strings:
                accumulated_str = "".join(accumulated_strings)
                yield ' ' * min(indent_level, max_indent_level) * indent_size + f'{accumulated_str}'
            # Get the current file position for debugging
            current_pos = self.file.tell()
            raise Exception(f"Error at file position {current_pos}: {str(e)}")
            # Don't re-raise the exception to allow partial output

    def read(self, byteorder: str = 'little') -> Any:
        """
        Read an xtype file and convert it to a Python object.

        This method is the counterpart to the write method. It reads the xtype file
        and returns a Python object that corresponds to what was written.

        Args:
            byteorder: The byte order of multi-byte integers in the file. Defaults to 'little'.

        Returns:
            Any: The Python object read from the file
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        # Reset the file position to the beginning
        self.file.seek(0)

        # Start recursive parsing
        return self._read_object(byteorder)

    def _read_object(self, byteorder: str = 'little') -> Any:
        """
        Read an object from the file.

        Args:
            byteorder: The byte order of multi-byte integers in the file.

        Returns:
            The Python object read from the file
        """
        # Get the next logical element
        symbol, dimensions, size = self._read_step(byteorder)
        # Process the symbol
        return self._read_element(symbol, dimensions, size, byteorder)

    def _read_element(self, symbol: str, dimensions: List[int], size: int, byteorder: str) -> Any:
        """
        Read an element based on its symbol from the file.

        Args:
            symbol: The symbol or type code read from the file
            dimensions: List of dimensions for array types (empty for scalar values)
            size: The size of binary data in bytes (0 for grammar symbols)
            byteorder: The byte order of multi-byte integers in the file

        Returns:
            The Python object read from the file
        """
        # Handle special symbols first
        if symbol == '[':
            # List
            return self._read_list(byteorder)
        elif symbol == '{':
            # Dictionary
            return self._read_dict(byteorder)
        elif symbol == 'T':
            # True
            return True
        elif symbol == 'F':
            # False
            return False
        elif symbol == 'n':
            # None
            return None
        elif symbol in self.type_sizes:
            # Check if this is an array type or a single element
            if dimensions:
                # This is an array type
                return self._read_numpy_array(dimensions, symbol, size, byteorder)
            else:
                # This is a single element
                return self._read_single_element(symbol, size, byteorder)
        else:
            # Unexpected symbol
            raise ValueError(f"Unexpected symbol in xtype file: {symbol}")

    def _read_single_element(self, type_code: str, size: int, byteorder: str) -> Any:
        """
        Read a basic element from the file.

        Args:
            type_code: The xtype type code
            size: The total size of binary data in bytes
            byteorder: The byte order of multi-byte integers in the file

        Returns:
            The element read from the file
        """
        # Read the binary data
        binary_data = self.read_raw_data(size)

        # Parse based on type code
        if type_code == 'b':
            # Boolean
            return binary_data[0] != 0
        elif type_code in 'ijkl':
            # Signed integers
            if type_code == 'i':
                return int.from_bytes(binary_data, byteorder=byteorder, signed=True)
            elif type_code == 'j':
                return int.from_bytes(binary_data, byteorder=byteorder, signed=True)
            elif type_code == 'k':
                return int.from_bytes(binary_data, byteorder=byteorder, signed=True)
            elif type_code == 'l':
                return int.from_bytes(binary_data, byteorder=byteorder, signed=True)
        elif type_code in 'IJKL':
            # Unsigned integers
            if type_code == 'I':
                return int.from_bytes(binary_data, byteorder=byteorder, signed=False)
            elif type_code == 'J':
                return int.from_bytes(binary_data, byteorder=byteorder, signed=False)
            elif type_code == 'K':
                return int.from_bytes(binary_data, byteorder=byteorder, signed=False)
            elif type_code == 'L':
                return int.from_bytes(binary_data, byteorder=byteorder, signed=False)
        elif type_code in 'hfd':
            # Floating point
            if type_code == 'h':
                # float16
                return struct.unpack('>e', binary_data)[0]
            elif type_code == 'f':
                # float32
                return struct.unpack('>f', binary_data)[0]
            elif type_code == 'd':
                # float64
                return struct.unpack('>d', binary_data)[0]
        elif type_code == 's':
            # String
            return binary_data.decode('utf-8')
        elif type_code == 'x':
            # Bytes
            return binary_data
        else:
            # Unsupported type
            raise ValueError(f"Unsupported type code: {type_code}")

    def _read_list(self, byteorder: str = 'little') -> List:
        """
        Read a list from the file.

        Args:
            byteorder: The byte order of multi-byte integers in the file.

        Returns:
            List: The list read from the file
        """
        result = []

        # Parse each element until we hit a closing bracket
        while True:
            symbol, dimensions, size = self._read_step(byteorder)

            if symbol == ']':
                # End of list
                break
            elif symbol in self.type_sizes:
                # Data type
                if dimensions:
                    # Array type
                    result.append(self._read_numpy_array(dimensions, symbol, size, byteorder))
                else:
                    # Basic element
                    result.append(self._read_single_element(symbol, size, byteorder))
            else:
                # Special symbol or container
                result.append(self._read_element(symbol, dimensions, size, byteorder))

        return result

    def _read_dict(self, byteorder: str = 'little') -> Dict:
        """
        Read a dictionary from the file.

        Args:
            byteorder: The byte order of multi-byte integers in the file.

        Returns:
            Dict: The dictionary read from the file
        """
        result = {}

        # Parse key-value pairs until we hit a closing brace
        while True:
            # Read the key
            symbol, dimensions, size = self._read_step(byteorder)

            if symbol == '}':
                # End of dictionary
                break

            # We're reading a key, which should be a string
            if symbol == 's':
                # String key
                key_binary = self.read_raw_data(size)
                key = key_binary.decode('utf-8')
            elif symbol == 'u':
                # String key
                key_binary = self.read_raw_data(size)
                key = key_binary.decode('utf-16')
            elif symbol in 'ijklIJKL':
                if dimensions:
                    # Int array type
                    intArray = self._read_numpy_array(dimensions, symbol, size, byteorder)
                    key = self._convert_to_deep_tuple(intArray.tolist())
                else:
                    # Int element
                    key = int(self._read_single_element(symbol, size, byteorder))
            elif symbol in 'hfd':
                if dimensions:
                    # Float array type
                    intArray = self._read_numpy_array(dimensions, symbol, size, byteorder)
                    key = self._convert_to_deep_tuple(intArray.tolist())
                else:
                    # Float element
                    key = float(self._read_single_element(symbol, size, byteorder))
            else:
                # Unexpected symbol for key
                raise ValueError(f"Unexpected key type in dictionary: {symbol}")

            # Read the value
            symbol, dimensions, size = self._read_step(byteorder)

            # We're reading a value
            if symbol in self.type_sizes:
                # Data type
                if dimensions and (symbol not in 'sx' or len(dimensions) > 1):
                    # Array type
                    result[key] = self._read_numpy_array(dimensions, symbol, size, byteorder)
                else:
                    # Basic element
                    result[key] = self._read_single_element(symbol, size, byteorder)
            else:
                # Special symbol or container
                result[key] = self._read_element(symbol, dimensions, size, byteorder)

        return result

    def _read_step(self, byteorder: str = 'little') -> Tuple[str, List[int], int]:
        """
        Read a single elementary part from the xtype file.

        This method reads the xtype file using read_raw and returns elementary parts.
        The elements could be single symbols ([]{}TFn*), scalar types or array types.
        The function does not read the binary payload data. This is done by calling
        read_raw_data().

        Args:
            byteorder: The byte order of multi-byte integers in the file. Defaults to 'little'.

        Returns:
            Tuple[str, List[int], int]: A tuple containing:
                - symbol: String representing the type or grammar symbol
                - dimensions: List of dimensions (empty for scalar values)
                - size: Size to be read in bytes (0 for grammar symbols without binary data)
        """
        if not self.file or self.file.closed:
            raise IOError("File is not open for reading")

        if self.mode != 'r':
            raise IOError("File is not open in read mode")

        # Store length values (dimensions) for array types
        dimensions = []

        # Process raw elements until we find a complete logical element
        for symbol, flag, length_or_size in self.read_raw(byteorder=byteorder):
            # Case 1: Grammar terminals (single symbols)
            if symbol in '[]{}TFn*':
                return symbol, [], 0

            # Case 2: Length information (0-9, M, N, O, P)
            elif flag == 1:
                # Store dimension for array types
                dimensions.append(length_or_size)
                continue

            # Case 3: Data types with binary data
            elif flag == 2:
                # Return the data type with collected dimensions and size
                return symbol, dimensions, length_or_size

        # If we reach here, we've reached the end of the file
        return '', [], 0

    def _read_numpy_array(self, dimensions: List[int], type_code: str, size: int, byteorder: str = 'little') -> np.ndarray:
        """
        Read a NumPy array from the file.

        Args:
            dimensions: The dimensions of the array
            type_code: The xtype type code
            size: The total size of binary data in bytes
            byteorder: The byte order of multi-byte integers in the file

        Returns:
            np.ndarray: The NumPy array read from the file
        """
        # Read the binary data
        binary_data = self.read_raw_data(size)

        # Special handling for string arrays
        if type_code == 's':
            # For 1D arrays, return a Python string
            if len(dimensions) == 1:
                # Decode the binary data as UTF-8 and return as a string
                return binary_data.decode('utf-8')
            else:
                # For multidimensional arrays, the last dimension is the string length
                string_length = dimensions[-1]
                array_dims = dimensions[:-1]

                # Calculate total number of strings
                total_strings = 1
                for dim in array_dims:
                    total_strings *= dim

                # Create a numpy array of fixed-length strings
                string_array = np.empty(array_dims, dtype=f'S{string_length}')

                # Fill the array with the strings from binary_data
                flat_array = string_array.reshape(-1)

                for i in range(total_strings):
                    start = i * string_length
                    end = start + string_length
                    if start < len(binary_data):
                        # Get the string data, ensuring we don't go past the end of binary_data
                        string_data = binary_data[start:min(end, len(binary_data))]
                        # Pad with zeros if needed
                        if len(string_data) < string_length:
                            string_data = string_data.ljust(string_length, b'\x00')
                        flat_array[i] = string_data

                return string_array.reshape(array_dims)

        # Map xtype type codes to NumPy dtypes
        dtype_map = {
            'b': np.bool_,
            'i': np.int8,
            'j': np.int16,
            'k': np.int32,
            'l': np.int64,
            'I': np.uint8,
            'x': np.uint8,
            'J': np.uint16,
            'K': np.uint32,
            'L': np.uint64,
            'h': np.float16,
            'f': np.float32,
            'd': np.float64
        }

        # Get the NumPy dtype
        if type_code not in dtype_map:
            raise ValueError(f"Unsupported NumPy type: {type_code}")

        dtype = dtype_map[type_code]

        # Calculate total number of elements
        total_elements = 1
        for dim in dimensions:
            total_elements *= dim

        # Create a flat array first
        if type_code == 'b':
            # Handle boolean arrays specially (0x00 for False, anything else for True)
            flat_array = np.frombuffer(binary_data, dtype=np.uint8)
            flat_array = flat_array != 0
        elif type_code in 'jklJKLhfd':
            # Signed integers (need to be byteswapped because xtype uses big-endian)
            flat_array = np.frombuffer(binary_data, dtype=dtype)
            flat_array = flat_array.byteswap()
        elif type_code in 'iIx':
            # Floating point (need to be byteswapped because xtype uses big-endian)
            flat_array = np.frombuffer(binary_data, dtype=dtype)
        else:
            # Unsupported type
            raise ValueError(f"Unsupported NumPy type: {type_code}")

        # Reshape the array to the specified dimensions
        return flat_array.reshape(dimensions)

    def _determine_data_size(self, type_code: str) -> Optional[int]:
        """
        Determine the size of data to read based on the type code.

        Args:
            type_code: The xtype type code

        Returns:
            Optional[int]: Size in bytes to read, or None if not a data type
        """
        # Single character type codes
        if type_code in self.type_sizes:
            return self.type_sizes[type_code]

        # Length indicators
        if type_code in 'MNOP':
            return {'M': 1, 'N': 2, 'O': 4, 'P': 8}[type_code]

        return None

    def _convert_to_deep_tuple(self, lst: List) -> Tuple:
        """
        Convert a list to a deep tuple.

        Args:
            lst: The list to convert

        Returns:
            Tuple: The deep tuple
        """
        if not isinstance(lst, list):
            return lst
        return tuple(self._convert_to_deep_tuple(i) for i in lst)
