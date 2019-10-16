# Universal Binary Notation (UBN)

Universal Binary Notation file format specification (beta)

Introduction
------------
Universal Binary Notation (UBN) is a general purpose self-explained binary file format for hierarchically structured data. The basic syntax is very simple and easy to implement but it tries to satisfy the needs of all imaginable applications for binary data storage and exchange. The simple grammar allows to generate typical data files by a few lines of code without any library. Simultaneously, with the use of optional metadata big data files can be generated and handled efficiently with random access.

The Vision
----------

The digital revolution is based on the simple convention to notate all kind of information by series of zeros and ones. The ASCII or unicode standard gives it a language that everybody can understand. Although the usefulness of such conventions is very obvious, it is missing at a higher level for binary data structures. Many thousand binary file formats exist in the world that need custom-built programs or libraries to decode them. As a consequence, many people use text formats for data exchange as it is the most universal standard. As a drawback text formats have limited speed and storage efficiency since numerical values have to be translated into their decimal text representations. Also certain sub elements cannot be read without parsing the whole text file.

This is where the vision of a Universal Binary Notation (UBN) comes into play:

* It should cover all desirable properties of binary formats.
* It should be very flexible and hierarchically structured.
* One single editor should be able to display the content of all binary formats based on it.
* Reading should be efficient and random access should be possible for sub-elements.

Existing solutions
------------------

For text files universal formats already exist, such as [XML](https://www.w3.org/XML/) and [JSON](http://www.json.org/) with the disadvantage of low efficiency. Existing universal binary format are e.g. [HDF5](https://www.hdfgroup.org/HDF5/) (Hierarchical Data Format), which is suitable for huge scientific data sets but has a quite complicated grammar and [UBJSON](https://github.com/ubjson/universal-binary-json/) (Universal Binary Java Script Object Notation), which is very simple, but is not optimized for big databases with random access. UBN is supposed to unify all needs.

Core Idea
---------

To unify the contradicting requirements of simplicity and advanced features, the format specification is expandable. The core grammar describes a minimalistic hierarchical data structure (inspired by UBJSON) while advanced features, as required for random access and handling of big data, are hidden in so-called _footnotes_. As in books footnotes give additional information about the content but they are not mandatory to understand the book. A parser for UBN is not required to understand and use the information inside the footnote. Most UBN files of typical usage will probably not require any footnotes and meta information.

UBN is also suitable for data streams. All elements of the grammar begin with ascii characters with values between 32 and 127 and have a defined end. Other ASCII values can be used for communication protocols.

### Features of the grammar

1. Basic boolean, integer, floating point data types and strings
2. Arrays, multi-dimensional arrays
3. Structured types
4. Lists of mixed arbitrary elements
5. Objects or dictionaries with key/value pairs for arbitrary elements
6. Arbitrary hierarchy levels
7. Unfinished files can be syntactically valid and complete as long as the last element of a list or dict is complete
8. Self-similarity: Inner elements can be extracted as complete and valid files

### Additional (possible) features that make use of the optional meta information in footnotes

1. Table of contents
2. Size information of elements
3. Fast random access to certain sub-elements in big files
4. Fast deletion and addition of elements in big files
5. Chunk data mode for efficient writing and reading of big files
6. Compressed elements
7. Included checksum
8. Definitions for date and time notation
9. Definitions for physical units

With the footnotes users have the freedom to create their own formats for their specific applications. This is similar to text formats where users can freely design file structures with their own format rules. But all text formats have in common that they are obliged to respect the ascii standard to make sure that every editor can show and edit the content. This idea is continued in UBN for binary formats. 

Status
------

UBN is under development. The grammar has still beta status and will be finalized at some point. There will be no different versions for the core grammar. The meta language, in contrast, is flexible and will grow from time to time and new features will be added.

Grammar (beta5)
--------------

The graphical representation of the grammar rules below should contain all information to enable a programmer writing valid UBN files. The red round boxes represent data to be written. Single black characters inside are stored directly as ASCII characters. Green boxes require nested grammer rules.

<p align="center"><img src="https://raw.githubusercontent.com/bitagoras/Universal-Binary-Notation/master/figures/UBN_file.png"></p>

<p align="center"><img src="https://raw.githubusercontent.com/bitagoras/Universal-Binary-Notation/master/figures/UBN_element.png"></p>

<p align="center"><img src="https://raw.githubusercontent.com/bitagoras/Universal-Binary-Notation/master/figures/UBN_list.png"></p>

<p align="center"><img src="https://raw.githubusercontent.com/bitagoras/Universal-Binary-Notation/master/figures/UBN_dict.png"></p>

<p align="center"><img src="https://raw.githubusercontent.com/bitagoras/Universal-Binary-Notation/master/figures/UBN_value.png"></p>

<p align="center"><img src="https://raw.githubusercontent.com/bitagoras/Universal-Binary-Notation/master/figures/UBN_type.png"></p>

<p align="center"><img src="https://raw.githubusercontent.com/bitagoras/Universal-Binary-Notation/master/figures/UBN_length.png"></p>

Other than in data of text formats no stop symbols can be defined for binary elements since the whole value range is reserved for the binary data. Therefore the size of the data must be determined by the parser from information stored in front of the data. The size of basic data types are given in the type table. In case of structs or arrays the number of bytes have to be added or multiplied accordingly.

## Types

| Type   | Name      | Bytes | Description                    | Comment                                       |
|:------:|-----------|:-----:|--------------------------------|-----------------------------------------------|
| `i, m` | uint8     | 1     | unsigned integer 8-bit         | C-type: unsigned char                         |
| `j, n` | uint16    | 2     | unsigned integer 16-bit        | C-type: unsigned short int                    |
| `k, o` | uint32    | 4     | unsigned integer 32-bit        | C-type: unsigned int                          |
| `l, p` | uint64    | 8     | unsigned integer 64-bit        | C-type: unsigned long int                     |
| `I`    | int8      | 1     | signed integer 8-bit           | C-type: char                                  |
| `J`    | int16     | 2     | signed integer 16-bit          | C-type: short int                             |
| `K`    | int32     | 4     | signed integer 32-bit          | C-type: int                                   |
| `L`    | int64     | 8     | signed integer 64-bit          | C-type: long int                              |
| `b`    | boolean   | 1     | boolean type                   | values: 0x00 = false or 0xFF = true           |
| `h`    | float16   | 2     | half precision float 16-bit    | IEEE 754-2008 half precission                 |
| `f`    | float32   | 4     | float 32-bit                   | IEEE 754 single precision, C-type: float      |
| `d`    | float64   | 8     | double precision float 64-bit  | IEEE 754 double precision, C-type: double     |
| `s`    | str/utf-8 | 1     | ascii / utf-8 string           | no other coding than utf-8 is specified       |
| `u`    | utf-16    | 2     | unicode string in utf-16       |                                               |
| `x`    | byte      | 1     | user defined data byte         | special structs, compressed data etc.         |
| `X`    | byte      | 1     | user defined data byte         | special structs, compressed data etc.         |

Examples
--------

In the examples below, characters in brackets `[ ]` symbolize characters that are directly stored as their ASCII values. Parentheses `( )` show readable representations of the corresponding binary data. If no type is noted for integers in parentheses the type is uint8. All examples are valid and complete UBN files. No additional header is required. That's simple, isn't it?

* **String**:

```
"hello world"

UBN:
     [m] (uint8: 11) [s] [h] [e] [l] [l] [o] [ ] [w] [o] [r] [l] [d]

hex: 6D          0B  73  68  65  6C  6C  6F  20  77  6F  72  6C  64
```

* **Integer:**

```
1025

UBN:
[j] (uint16: 1025)

hex: 6A 01 04
```

* **3d vector of type uint8:**

```
[10, 200, 255]

UBN:
       [3] [i] (uint8: 10) (uint8: 200) (uint8: 255)

hex:   33  69          0A           C8           FF   
```

* **List with integer, string and float:**

```
[7, "seven", 7.77]

UBN:
[[] [i] (uint8: 7) (uint8: 5) [s] [seven] [d] (float64: 7.77) []]
```

* **Struct with integer, string and float:**

```
[(uint8) 7, (string*5) "seven", (double) 7.77]

UBN:
[(] [i] [5] [s] [d] [)] (uint8: 7) [seven] (float64: 7.77)
```

* **3 x 3 matrix of double:**

```
[ [1.1, 3.3, 5.5],
  [2.2, 4.4, 6.6],
  [3.3, 5.5, 7.7] ]
  
UBN:  
[3]
    [3] [d]
        (float64: 1.1) (3.3) (5.5)
        (2.2) (4.4) (6.6)
        (3.3) (5.5) (7.7)
```

* **800 x 600 x 3 RGB Image:**

```
UBN:
[n] (uint16: 800)
    [n] (uint16: 600)
        [3] [i]
        (... 800*600*3 bytes of data ...)
```

* **Object:**

```
{
  "planet": "Proxima b",
  "mass": 1.27,
  "habitable": True
}

UBN:
[{]
    [6] [s] [planet] [9] [s] [Proxima b]
    [4] [s] [mass] [d] (float64: 1.27)
    [9] [s] [habitable] [T]
[}]
```

* **3 x 3 table of doubles with named colums "lon", "lat", "h":**

```
[ ["lon", "lat",   "h"],
  [1.1,    3.3,    5.5],
  [2.2,    4.4,    6.6],
  [3.3,    5.5,    7.7],
  [4.4,    6.6,    8.8] ]

UBN:
[[]
    [[]
        [3] [s] [lon]
        [3] [s] [lat]
        [s] [h]
    []]
    [4] [3] [d]
        (1.1) (3.3) (5.5)
        (2.2) (4.4) (6.6)
        (3.3) (5.5) (7.7)
        (4.4) (6.6) (8.8)
[]]
```

# Footnote Meta Language
## Overview

The content of the `footnote` element gives information and hints about how to read, interpret or pre-process the data, before it is passed to the application. A parser that makes no use of the footnotes has to parse the element after `[*]` but can ignore its content. The content is mostly used for optimizing the efficiency or speed for writing and reading large files with random access. It can also contain application-specific information of how to apply the data. The footnote can be a string or other data types. There is e.g. information with instructions to transpose or concatenate matrices or vectors when loaded into memory. The footnote allows to extend the UBN format without changing the core grammar.

All information about sizes or relative jump positions are related to the whole element including the footnote itself. So the parser has to remember the position of the `*` character of the footnote as the reference position. Also some zero-bytes belong to the element as defined in the grammar rule for the element and therefore is addressed by the footnote. 

Objects with footnotes can be nested. This is usefull for several footnote elements with different types, e.g.:
```
[*] (footnote with size) [*] (footnote with table of content) (data of type list)
```

footnote elements can have string identifiers to indicate the purpose of the footnote or to identify other use-case specific meta languages. The string identifiers are footnotes of the footnote:

```
[*] [*] (string "size") (footnote with size) [*] [*] (string "TOC") (footnote with table of content) (data of type list)
```

Use-case specific and user-defined footnotes have the format

```
[*] [*] (string with meta language identifier) [{] (dict with meta information) [}] (element)
```

If there is for example a binary format for numpy with specific information about the data type, the footnote would be

```
[*] [*] [numpy] [{] (dict with meta information) [}] (element)
```

## Default Footnote Meta Language

Version: 0.2

### Size of element

**Purpose:** Gives a size information about an element. 

**Optional identifier keyword:** `size`

**footnote type:** unsigned integer (`i`,`j`,`k`,`l`)

**footnote element value:** number of bytes of the whole data element, including this footnote

**Explanation:**

This footnote tells the number of bytes of an element. The size also includes the footnote itself, the `*` and spaces after the footnote. The size information helps to browse more quickly through the file structure in order to access a certain sub-element in large files, without parsing all previous elements.

**Example:**

Let's assume the element, without the size of the footnote, is 1200 byte. The footnote (with size 4 byte) would be:

```
[*] [j] (uint16: 1204) (data with 1200 byte)
```

Self-explained with footnote identifier keyword `size`:

```
[*] [*] [4] [s] [size] [j] (uint16: 1211) (data with 1200 byte)
```

### Deleted element

**Purpose:** Flags an element as deleted

**Optional identifier keyword:** `deleted`

**Footnote type:** None

**Footnote value:** `N` (None)

**Explanation:**

This footnote tags an element as deleted. This is useful for big files when an element in the middle should be deleted without rewriting the whole file. Small elements can be deleted by overwriting them with zero-bytes. For larger elements footnotes like this can be added, followed by an `x` array that covers the element until the end. By this a very large element can be deleted by writing only a few bytes at the beginning. Next time the entire file is rebuilt, the unused space can be eliminated.

**Example:**

In the following example an element with 10000 bytes is tagged as deleted. The included footnote and the `x` byte-array type definition together are 6 bytes long. The remaining bytes of the 10000 bytes are covered by the 9994 long `x` array. So, only 6 bytes have to be written to remove the element, instead of writing 10000 zero bytes or rebuilding the whole file which may require to update some links in the table of contents.

```
[*] [N]
[n] (uint16: 9994) [x] (data with 9994 byte)
```

### Element visibility

**Purpose:** Flags an element as visible or invisible (disabled)

**Optional identifier keyword:** `enabled`

**Footnote type:** boolean `T` or `F`

**Footnote value:** `T` (true for enabled), `F` (false for disabled or deleted)

**Explanation:**

This footnote type tags an element as invisible, when the value is set to false. This feature can be used for reserving some space for e.g. elements to be added later or as placeholder for a table of content that will be added after all subelements are written and their sizes and positions are known.

**Example:**

In the following example an element is tagged as invisible. This element is treated as non-exisiting, but the element will not be deleted when the file is rebuilt. This could be an element that is inserted later at another list by a certain link instruction but is physically added at the end for efficiency reasons. Note that invisible elements can only be placed at locations where the grammar allows an element.

```
[*] [F] (some element)
```

## Table of content for quick random access

**Purpose:** Table of content: List with the relative starting positions of all elements in a list or dict data

**Optional identifier keyword:** `TOC`

**Footnote type:** array of unsigned integer (`i`,`j`,`k`,`l`)

**Footnote value:** relative byte offset to the list elements from the beginning of the footnote

**Explanation:**

This footnote type allows to access elements of lists or dicts in large data files. The relative offsets are stored in an integer array with the same length as the list or dict object. The offset points to the beginning of each element (in list) or the keyword value (in dict). If the targeting element has another footnote, the offset points to the `*` token which is the first byte of the list element.

**Example:**

This example shows short list with mixed types and a table of content with offsets

```
[7, "seven", 7.77]

UBN:

[*] [3] [i]                # uint8 array of length 3
        (7) (9) (16)       # offsets to the elements
[[] [i] (uint8: 7) (uint8: 5) [s] [seven] [f] (float32: 7.77) []]
     ^              ^                      ^   # target positions 
```

## Element links

**Purpose:** In more complex data structures and big files it can be usefull to use links to elements. This allows to quickly parse the main structure without reading the whole data and to allows to add, move or delete elements.

**Footnote type:** String

**Footnote value:** `@`

**Element value:** The content of the element is replaced by an unsigned integer (`i`,`j`,`k`,`l`) or array of unsigned integers that points to the absolute address (relative to the beginning of the file) of the elements with the actual data.

**Explanation:**

This footnote type allows to keep the main data structure small and efficient and allows fast random access to sub elements and more flexibility. The elements of the structure contain only the links to the actuall elements which are stored in a list at then end.

**Example:**

In this example a data structure contains some very big elements:

```
{
  'file1': 'bigdata1',
  'folder1': {'fileA': 'bigdata2', 'fileB': 'bigdata3'}
}

UBN:

[[]  # List
    # Data Structure with links instead of actual data elements
    [{]
        [5] [s] [file1]
            [*] [s] [@] [i] (...)  # Link to element bigdata1
        [{] 
            [5] [s] [fileA]
                [*] [s] [@] [i] (...)  # Link to element bigdata2
            [5] [s] [fileB]
                [*] [s] [@] [i] (...)  # Link to element bigdata3
            [ ] [ ] [ ] [ ] [ ] [ ] ...  # Place holder for adding elements in future
        [}]
    [}]
    [8] [s] [bigdata1]
    [8] [s] [bigdata2]
    [8] [s] [bigdata3]

# No []] at the end to append more elements in future

```
