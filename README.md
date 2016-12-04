# Universal Binary Notation (UBN)

Universal Binary Notation file format specification (beta)

Introduction
------------
Universal Binary Notation (UBN) is a general purpose self-explained binary file format for hierarchically structured data. The basic syntax is very simple and easy to implement but it tries to satisfy the needs of all imaginable applications for binary data storage and exchange. The simplicity allows to generate typical data files by a few lines of code without any library. Simultaneously, with the use of meta information big data files can be generated that are manageable efficiently with random access.

The Vision
----------

The success of the Digital Revolution is based on the common notation of all kind of information by binary series of zeros and ones. It is therefore all the more astonishing that the idea of unification did not find it's way to the syntax of binary data structures. Many thousands binary file formats exist in the world that need custom-built programs or libraries to decode the data. The leak of a unified data structure format counteracts the full advantage of digitalization.

For text files universal formats exist, like [XML](https://www.w3.org/XML/), [JSON](http://www.json.org/), [CSV](https://en.wikipedia.org/wiki/Comma-separated_values). As a drawback text formats have limited speed and storage efficiency since numerical values have to be translated into its decimal text representations and included elements cannot be read without parsing the whole text file. 

Here comes the vision of a Universal Binary Notation (UBN) that is easy to use and provides all desirable properties of binary formats. One single editor should be able to view and edit the content of any binary file format that is based on UBN. This would allow binary files to gain the popularity of text files, which can all be opened by one text-editor. Due to it's binary structure certain sub-elements can be accessed very efficiently. This would make binary files even more flexible than text-files and enable users to handle elements as intuitive as files in a directory tree of a file system.

But why again a new format? Does no common binary formats exist for general purposes? There are some examples, however they suffer from too high complexity or limited versatility. Examples are [HDF5](https://www.hdfgroup.org/HDF5/) (Hierarchical Data Format) and [UBJSON](https://github.com/ubjson/universal-binary-json/) (Universal Binary Java Script Object Notation). The former is feature-rich and suitable for huge scientific data sets but has a quite complicated grammar while the latter is very simple, but is not optimized for big databases. UBN is supposed to bridging the gap.

Core Idea
---------

To unify the contradicting requirements of simplicity and advanced features, the format specification is divided into two meta levels. The core grammar describes a very simple but hierarchical data structure. Advanced features such as random access are hidden in a special grammar rule for meta information. When ignoring the meta information, the file still can be parsed with the core grammar and most of the binary data types can be understood. It is assumed that more than 90% of the UBN files will not make use of the additional meta information.

### Features of the grammar

1. Basic boolean, integer, floating point data types and strings
2. Arrays, multi-dimensional arrays
3. Lists of mixed arbitrary elements
4. Objects or dictionaries with key/value pairs for arbitrary elements
5. Arbitrary hierarchy levels
6. Files are syntactically valid even if not all elements are written yet

### Possible features by making use of the optional meta information

1. Table of contents
2. Fast random access to single elements even in big files
3. Structured types
4. Fast deletion and addition of elements in big files
5. Chunk data mode for efficient writing and reading of big files
6. Compression of data
7. Included checksum
8. Datetime definition


Status
------

UBN is under development. The grammar will be finalized at some point when it is consollidated that nothing important is missing. There will be no different versions for the core grammar. At the moment a flag for a beta status is set. The meta language, in contrast, will grow from time to time and new features will be added. A version number will show it's status.

Grammar (beta1)
--------------

The graphical representation of the grammar rules below should enable a programmer to write valid UBN files. All red round boxes represent data to be written. Single black characters inside the round boxes are stored directly as ascii characters. Green boxes represent nested grammer rules.

<p align="center"><img src="https://github.com/bitagoras/Universal-Binary-Notation/blob/master/UBN_file.png?raw=true"></p>

<p align="center"><img src="https://github.com/bitagoras/Universal-Binary-Notation/blob/master/UBN_element.png?raw=true"></p>

<p align="center"><img src="https://github.com/bitagoras/Universal-Binary-Notation/blob/master/UBN_metainfo.png?raw=true"></p>

<p align="center"><img src="https://github.com/bitagoras/Universal-Binary-Notation/blob/master/UBN_list.png?raw=true"></p>

<p align="center"><img src="https://github.com/bitagoras/Universal-Binary-Notation/blob/master/UBN_dict.png?raw=true"></p>

<p align="center"><img src="https://github.com/bitagoras/Universal-Binary-Notation/blob/master/UBN_value.png?raw=true"></p>

<p align="center"><img src="https://github.com/bitagoras/Universal-Binary-Notation/blob/master/UBN_type.png?raw=true"></p>

<p align="center"><img src="https://github.com/bitagoras/Universal-Binary-Notation/blob/master/UBN_count.png?raw=true"></p>

In comparison to text files no stop symbol can be defined for binary elements since the whole value range is reserved for the binary data. Therefore the size of the data must be calculated and stored in front of the data. The sizes of the basic types are given in the type table of the grammar. In case of arrays the number of bytes have to be multiplied accordingly.

Examples
--------

In the examples below symbols in brackets ```[ ]``` denote ascii characters that are stored directly in the binary data. Parentheses ```( )``` show some human readable representation of the corresponding binary data. All examples are valid and complete UBN files. No additional header is required. That's simple, isn't it?

* **String**:

```
"hello world"

UBN:
(uint8: 11) [s] [h] [e] [l] [l] [o] [ ] [w] [o] [r] [l] [d]
```

* **3d vector of type uint8:**

```
[10, 200, 255]

UBN:
(uint8: 3) [I] (uint8: 10) (uint8: 200) (uint8: 255)
```

* **3 x 3 matrix of double:**

```
[ [1.1, 3.3, 5.5],
  [2.2, 4.4, 6.6],
  [3.3, 5.5, 7.7] ]
  
UBN:  
(uint8: 3)
    (uint8: 3) [d]
        (float64: 1.1) (3.3) (5.5)
        (2.2) (4.4) (6.6)
        (3.3) (5.5) (7.7)
```

* **800 x 600 x 3 RGB Image:**

```
UBN:
[N] (uint16: 800)
    [N] (uint16: 600)
        (uint8: 3) [I]
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
    (6) [s] ("planet") (9) [s] ("Proxima b")
    (4) [s] ("mass") [d] (float64: 1.27)
    (9) [s] ("habitable") [T]
[}]
```

* **3 x 3 table of doubles with named colums "lon", "lat", "alt":**

```
[ ["lon", "lat", "alt"],
  [1.1,    3.3,    5.5],
  [2.2,    4.4,    6.6],
  [3.3,    5.5,    7.7],
  [4.4,    6.6,    8.8] ]

UBN:
[[]
    (int8: 3)
        (int8: 3) [s]
            [l] [o] [n]
            [l] [a] [t]
            [a] [l] [t]
    (uint8: 3) [d] (1.1) (3.3) (5.5)
    (uint8: 3) [d] (2.2) (4.4) (6.6)
    (uint8: 3) [d] (3.3) (5.5) (7.7)
    (uint8: 3) [d] (4.4) (6.6) (8.8)
[]]
```

# Meta Language

Version: 0.1

The content of the ```metainfo``` object gives information about how to read or pre-process the data, before it is passed to the application. This is mostly used for optimizing the storage efficiency or speed for large files. It can also contain application-specific information of how to apply the data. The metainfo consists of objects in front of elements and contains e.g. pointers to sub-elements, definitions of user-defined types for ```x```, or instructions to transpose or concatenate matrices or vectors when loaded into memory. The metainfo object contains pairs of keywords and values. Each pair represents an extended feature that is not contained in the UBN core grammar. Some first examples of meta features are listed below. More will follow.

### Size of element

**Keyword:** ```size```

**Value:** ```n```

**With:**

```
   n: unsigned integer
      Number of bytes of element
```

**Explanation:**

This meta feature tells the number of bytes of an element. The size also includes the metainfo itself, as well as white-spaces after the metainfo. The size information helps to browse more quickly through the file structure in order to access a certain sub-element in large files, without parsing all previous elements.

**Example:**

Let's assume the element, including the meta information, is 1200 byte. The meta information would be:

```
[<] (4) [s] [i] [z] [e]
    [N] (1200) [x]
[>]
```

### Deleted element

**Keyword:** ```deleted```

**Value:** ```T``` (true) or ```F``` (false)

**Explanation:**

This meta feature tags an element as deleted, wthen the value is set to true. This is usefull for big files when an element in the middle should be deleted without rewriting the whole file. Small elements can be deleted by overwriting them with spaces. For larger elements a metainfo like this can be added, followed by an ```x``` array that covers the element until the end. By this a very large element can be deleted by writing only a few bytes at the beginning. Next time the entire file is rebuit, the unused space can be discarded. This feature also can be used to reserve some space for e.g. a table of content that will be included later.

**Example:**

In the following example an element with 10000 bytes is taged as deleted. The included metainfo and the ```x``` array type definition are together 15 bytes long. The remaining bytes of the 10000 bytes are covered by the 9985 long ```x``` array. So, only 15 bytes had to be written to remove the element, instead of writing 10000 spaces or rebuiding the whole file.

```
[<] (7) [d] [e] [l] [e] [t] [e] [d]
    [T]
[>]
[N] (9985) [x]
```

### Structured types

**Keyword:** ```struct```

**Value:** ```{ (ax: v1): (ax: v2), ...}```

**With:**

```
   ax: array of type x or y
   
   v1: (array of type x or y definition)
       Bytes that match a type x or y array definition
       
   v2: (type) (type) ... 
       List of types that form the struct
```
**Explanation:**

The value of the meta feature keyword ```struct``` is of type ```dict```. The dict contains one or multiple struct definitions. Each key of the dict is e.g. the ```x```-type including a certain length (```v1```) and the value of the dict contains the type definition of the struct (```v2```). A key could be for example ```v1 = (12) [x]```. This means that all arrays of type ```x``` that have the exact length of 12 will be interpreted as the struct defined by ```v2```. A struct definition that consist of an int32 and a float64 would be ```v2 = [k] [d]```. Both ```v1``` and ```v2``` are of type ```x```.

In case were many structs of the same size must be defined, ```v1``` can also contain one or two data bytes (type uint8 or uint16) as identifiers to distinguish the different structs. The same ID must then be present at the beginning of every data of the ```x``` or ```y``` array containing the structured data. The struct definition will refer only to the part of the ```x``` or ```y``` array with the structured data, excluding the struct ID bytes.

The struct definition is only valid for the related element and all sub-elements. Nested elements can overwrite outer struct definitions temporarily inside the own scope.

**Example 1:**

Let's assume we want to define a struct int8 + float32. The metainfo would be

```
[<] (6) [s] [t] [r] [u] [c] [t]
    [{]
        (2) [x]
            (5) [x]
        (2) [x]
            [i] [f]
    [}]
[>]
```

An element of the struct with e.g. the numbers 120 and 2.25:

```
(2) [x] (int8: 123) (float32: 2.25)
```

When the interpreter finds a type that matches ```(2) [x]```, the bytes are interpreted as the defined struct.

**Example 2:**

Let's assume we want to define a struct int8 + float32 and another one int16 + int16 + int8, both of same size (5 bytes). The two types ```y``` and ```x``` could be used to distinguish in this case two structs. When many more structs have to be defined of same length, the following metainfo can be used:

```
[<] (6) [s] [t] [r] [u] [c] [t]
    [{]
        (3) [x]
            (6) [x] (55)
        (2) [x]
            [i] [f]
        (3) [x]
            (6) [x] (77)
        (3) [x]
            [j] [j] [i]
    [}]
[>]
```
The values (55) and (77) are arbitrary IDs to distinguish the two structs. 

An element of the first struct (ID 55) with the data 120 and 2.25 would be:

```
(6) [x] (55) (int8: 123) (float32: 2.25)
```

An element of the second struct (ID 77) with the data 1234, 2345 and 11 would be:

```
(6) [x] (77) (int16: 1234) (int16: 2345) (int8: 11)
```
