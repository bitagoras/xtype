# Universal Binary Notation (UBN)

Universal Binary Notation file format specification (beta)

Introduction
------------
Universal Binary Notation (UBN) is a general purpose self-explained binary file format for hierarchically structured data. The basic syntax is very simple and easy to implement but it tries to satisfy the needs of all imaginable applications for binary data storage and exchange. The simple grammar allows to generate typical data files by a few lines of code without any library. On the other hand more complex files and huge data bases can be defined and handled efficiently with random access.

The Vision
----------

The digital revolution is based on the simple convention of noting all kinds of information through a series of zeros and ones. Although the usefulness of such conventions is very obvious, it is missing at a higher level for binary data structures. Too many binary file formats exist that need custom-built programs or libraries to decode them. As a consequence, many people use text formats for data exchange as it is the most universal standard. As a drawback text formats have limited speed and storage efficiency since numerical values have to be translated into their decimal text representations. Also certain sub elements cannot be read without parsing the whole text file.

This is where the vision of the format Universal Binary Notation (UBN) comes in:

* It should cover all needs of binary formats.
* It should be very flexible and hierarchically structured.
* It should be so simple without overhead that writing simple files do not require any library.
* One single editor should be able to display the content of all binary formats based on it.
* Reading should be efficient and random access should be possible for sub-elements.

Existing solutions
------------------

For text files universal formats already exist, such as [XML](https://www.w3.org/XML/) and [JSON](http://www.json.org/) with the disadvantage of low efficiency. For universal binary formats one can mention e.g. [HDF5](https://www.hdfgroup.org/HDF5/) (Hierarchical Data Format), which is suitable for huge scientific data sets but has a quite complicated grammar. Much more simpler is [UBJSON](https://github.com/ubjson/universal-binary-json/) (Universal Binary Java Script Object Notation), but is not as versatile and cannot be used for big and complex databases with random access. UBN is supposed to enable all desirable properties of text and binary formats.

The Philosophy
--------------

To unify the contradicting requirements of simplicity and advanced features, the format specification is expandable. The simple grammar describes a minimalistic hierarchical data structure (inspired by UBJSON) while advanced features, as required for random access and handling of big data, are optional and hidden in so-called _footnotes_ without blowing up the grammar. Just like you know it from books, footnotes give additional information about the content, but they are not mandatory to understand the book. A parser for UBN is not required to understand and use the information inside the footnote to read the elements of the file. Most UBN files of typical usage will probably not require any footnotes at all.

UBN is also suitable for data streams. All elements of the grammar begin with ascii characters with values between 32 and 127 and have a defined end. Other ASCII symbols can be used before or after the elements for communication protocols.

Properties
----------

### Features of the grammar

1. Basic boolean, integer, floating point data types and strings
2. Arrays, multi-dimensional arrays
3. Structured types
4. Lists of mixed arbitrary elements
5. Objects or dictionaries with key/value pairs for arbitrary elements
6. Arbitrary hierarchy levels
7. Logging: Files with unfinished lists are syntactically valid which allows to append more elements later.
8. Self-similarity: Inner elements can be extracted and stored independently as complete and valid files.

### Additional (possible) features that make use of the optional meta information in footnotes

1. Table of contents
2. Fast random access to certain sub-elements in big files
3. Fast deletion and addition of elements in big files
4. Chunk data mode for efficient writing and reading of big files
5. Included checksum
6. Definitions for date and time notation
7. Definitions for physical units
8. Compressed elements

With the footnotes users have the freedom to create their own formats for their specific applications. This is similar to text formats where users can freely design file structures with their own format rules. But all text formats have in common that they respect the ascii standard to make sure that every editor can show and edit the content. This idea is used by UBN and continued for binary formats. 

Status
------

UBN is under development. The grammar has still beta status and will be finalized at some point. There will be no different versions for the core grammar. The meta language of the footnotes, in contrast, is flexible and can grow from time to time and new features will be added or conventions can be changed while outdated files still can be parsed.

Grammar (beta6)
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
| `e`    | element   | 1     | element as defined in grammar  | Encapsulated element in array of e            |
| `x`    | byte      | 1     | user defined data byte         | special structs, compressed data etc.         |

A special type is `e` for elements which can be used as an array of bytes to reserve space for another UBN element. It acts as an additional size information for elements. It helps to parse a file more quickly and step over larger subelements.

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

* **4 x 3 table of doubles with named colums "lon", "lat", "h":**

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

The content of the `footnote` element gives information and hints about how to read, interpret or pre-process the data, before it is used by the application. A parser that makes no use of the footnotes must parse the element after `[*]` to find out its size but can ignore its content. The content is mostly used for optimizing the efficiency or speed for writing and reading large files with random access. It can also contain application-specific information on how to apply the data. The footnote can be a string or any other data type.

Information about jump positions in table of contents are given relative to the `*` character of the footnote as the reference position.

Objects with several footnotes can be nested:
```
[*] (footnote with unit) [*] (footnote with table of content) (data of type list)
```

Several footnotes can also be listed:
```
[*] [[] (footnote with unit) (footnote with table of content) []] (data of type list)
```

Footnote elements can have string identifiers to indicate the purpose of the footnote or to identify use-case specific meta languages. The string identifiers are footnotes of the footnote:

```
[*] [*] (string "unit") (string "cm") [*] [*] (string "TOC") (footnote with table of content) (data of type list)
```

Use-case specific and user-defined footnotes are recommended have the format

```
[*] [*] (string with meta language identifier) [{] (dict with meta information) [}] (element)
```

If there is for example a binary format for `numpy` with specific information about the data type, the footnote would be

```
[*] [*] [numpy] [{] (dict with meta information) [}] (element)
```

## Default Footnote Meta Language

Version: 0.2

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
        [}]
        [*] [F] [n] (1000) [x] (1000 Byte) # Invisible place holder buffer for
    [}]                                    # adding more elements in future
    [8] [s] [bigdata1]
    [8] [s] [bigdata2]
    [8] [s] [bigdata3]

# No []] at the end to append more elements in future

```
