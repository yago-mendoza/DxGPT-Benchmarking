# ICD-10 Taxonomy API

A minimalist, zen-like Python API for working with ICD-10 medical classification codes. Fast, elegant, and comprehensive hierarchical analysis.

## Features

- **Universal Input**: All methods work with both codes AND names
- **Case Insensitive**: Works with any case combination
- **Lazy Loading**: Data loads only when needed
- **Smart Caching**: Fast repeated operations
- **Clean API**: Simple, memorable method names
- **Flexible Output**: Format parameter for code, name, or both

## Installation

```python
from utils.icd10 import ICD10Taxonomy
t = ICD10Taxonomy()
```

## API Reference

### 1. Basic Information Retrieval

#### `get(code_or_name)` → dict
Get basic information about any code or name.

```python
>>> t.get('S62')
{'code': 'S62', 'name': 'Fracture at wrist and hand level', 'type': 'category', 
 'full_key': 'S62 Fracture at wrist and hand level'}

>>> t.get('cholera')  # Works with names too!
{'code': 'A00', 'name': 'Cholera', 'type': 'category', 'full_key': 'A00 Cholera'}
```

#### `fullkey(code_or_name)` → str
Get full key (code + name) for any input.

```python
>>> t.fullkey('S62')
'S62 Fracture at wrist and hand level'

>>> t.fullkey('cholera')
'A00 Cholera'
```

#### `shift(code_or_name)` → str
Convert between codes and names bidirectionally.

```python
>>> t.shift('S62')  # Code → Name
'Fracture at wrist and hand level'

>>> t.shift('Cholera')  # Name → Code
'A00'

>>> t.shift('Fracture at wrist and hand level')  # Name → Code
'S62'
```

#### `type(code_or_name)` → str
Get the hierarchy type (chapter, range, category, block, subblock, group, subgroup).

```python
>>> t.type('XIX')
'chapter'

>>> t.type('S60-S69')
'range'

>>> t.type('cholera')  # Works with names
'category'
```

### 2. Search Function

#### `match(query, exact=False, type=None, format='code')` → list
Search for codes by name. Case insensitive. 
- `exact`: If True, only exact name matches
- `type`: Filter by hierarchy type ('chapter', 'range', 'category', 'block', 'subblock', 'group', 'subgroup')
- `format`: Output format ('code', 'name', 'both')

```python
>>> t.match('cholera', exact=True)
['A00']

>>> t.match('fracture')[:5]  # Partial match
['K08.53', 'K08.530', 'K08.531', 'K08.539', 'M48.40XA']

>>> t.match('injury', format='name')[:3]
['Injury of muscle and tendon of long head of biceps', 
 'Injury of muscle and tendon of other parts of biceps', 
 'Injury of muscle and tendon of triceps']

>>> t.match('injury', format='both')[:2]
['S46.10 Injury of muscle and tendon of long head of biceps', 
 'S46.20 Injury of muscle and tendon of other parts of biceps']

# Filter by hierarchy type
>>> t.match('injury', type='category')[:3]  # Only categories
['S14', 'S24', 'S34']

>>> t.match('fracture', type='block')[:3]  # Only blocks
['S02.0', 'S02.1', 'S02.2']

>>> t.match('fracture', type='category', format='both')[:2]
['S02 Fracture of skull and facial bones',
 'S12 Fracture of cervical vertebra and other parts of neck']

>>> len(t.match('disease', type='chapter'))  # Count chapters with 'disease'
8

>>> t.match('infectious', type='range', format='name')[:2]
['Intestinal infectious diseases', 'Certain zoonotic bacterial diseases']
```

### 3. Hierarchy Navigation

#### `children(code_or_name, type=None, format='code')` → list
Get children of a code. Type parameter options:
- `None` (default): All descendants
- Integer: Depth level (0=immediate, 1=next level, etc.)
- String: Specific hierarchy type ('block', 'group', etc.)

```python
>>> t.children('S62', type=0)  # Immediate children only
['S62.0', 'S62.1', 'S62.2', 'S62.3', 'S62.4', 'S62.5', 'S62.6', 'S62.8', 'S62.9']

>>> t.children('S62', type='block')  # All blocks under S62
['S62.0', 'S62.1', 'S62.2', 'S62.3', 'S62.4', 'S62.5', 'S62.6', 'S62.8', 'S62.9']

>>> t.children('cholera')  # All descendants (default)
['A00.0', 'A00.1', 'A00.9']

>>> len(t.children('XIX'))  # Count all descendants
11370

>>> t.children('S62', type=0, format='name')[:3]
['Fracture of navicular [scaphoid] bone of wrist', 
 'Fracture of other and unspecified carpal bone(s)', 
 'Fracture of first metacarpal bone']
```

#### `parent(code_or_name, format='code')` → str
Get immediate parent.

```python
>>> t.parent('S62')
'S60-S69'

>>> t.parent('S62.3', format='name')
'Fracture at wrist and hand level'

>>> t.parent('S62.3', format='both')
'S62 Fracture at wrist and hand level'
```

#### `parents(code_or_name, format='code')` → list
Get all parents up to root (bottom-up order).

```python
>>> t.parents('S62.302A')
['S62.302', 'S62.30', 'S62.3', 'S62', 'S60-S69', 'XIX']

>>> t.parents('S62.302A', format='both')[:2]
['S62.302 Unspecified fracture of third metacarpal bone', 
 'S62.30 Unspecified fracture of other metacarpal bone']
```

#### `path(code_or_name, format='code')` → list
Get full path from root to code (top-down order).

```python
>>> t.path('S62.3')
['XIX', 'S60-S69', 'S62', 'S62.3']

>>> t.path('S62.3', format='name')
['Injury, poisoning and certain other consequences of external causes', 
 'Injuries to the wrist and hand', 
 'Fracture at wrist and hand level', 
 'Fracture of other and unspecified metacarpal bone']

>>> t.path('S62.3', format='both')
['XIX Injury, poisoning and certain other consequences of external causes',
 'S60-S69 Injuries to the wrist and hand',
 'S62 Fracture at wrist and hand level',
 'S62.3 Fracture of other and unspecified metacarpal bone']
```

#### `siblings(code_or_name, format='code')` → list
Get siblings of the same type at the same level.

```python
>>> t.siblings('S62')[:5]  # First 5 siblings (all categories)
['S60', 'S61', 'S63', 'S64', 'S65']

>>> t.siblings('S62', format='name')[:3]
['Superficial injury of wrist, hand and fingers', 
 'Open wound of wrist, hand and fingers', 
 'Dislocation and sprain of joints and ligaments at wrist and hand level']

>>> len(t.siblings('S62'))  # Count siblings
9
```

### 4. Complete Analysis

#### `hierarchy(code_or_name)` → dict
Get complete hierarchical information in one call.

```python
>>> t.hierarchy('S62.3')
{
    'code': 'S62.3',
    'name': 'Fracture of other and unspecified metacarpal bone',
    'type': 'block',
    'full_key': 'S62.3 Fracture of other and unspecified metacarpal bone',
    'parents': {
        'category': {'code': 'S62', 'name': 'Fracture at wrist and hand level'},
        'range': {'code': 'S60-S69', 'name': 'Injuries to the wrist and hand'},
        'chapter': {'code': 'XIX', 'name': 'Injury, poisoning...'}
    },
    'children': {
        'subblock': 10,
        'group': 10,
        'subgroup': 200
    },
    'n_children': 10,  # Immediate children count
    'n_siblings': 8,   # Siblings of same type
    'path': ['XIX', 'S60-S69', 'S62', 'S62.3']
}
```

### 5. Type-Specific List Methods

Get all codes of a specific type with optional formatting.

```python
>>> t.chapters()  # All chapter codes
['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', ...]

>>> t.chapters(format='name')[:3]
['Certain infectious and parasitic diseases', 
 'Neoplasms', 
 'Diseases of the blood and blood-forming organs...']

>>> t.ranges()[:5]  # First 5 range codes
['A00-A09', 'A15-A19', 'A20-A28', 'A30-A49', 'A50-A64']

>>> t.categories()[:5]  # First 5 category codes
['A00', 'A01', 'A02', 'A03', 'A04']

>>> len(t.blocks())  # Count all blocks
9916

>>> len(t.groups())  # Count all groups
18241
```

Available methods:
- `chapters(format='code')`
- `ranges(format='code')`
- `categories(format='code')`
- `blocks(format='code')`
- `subblocks(format='code')`
- `groups(format='code')`
- `subgroups(format='code')`

## Counting with len()

Use Python's built-in `len()` function for all counting operations:

```python
# Count all chapters
>>> len(t.chapters())
22

# Count groups within chapter XIX
>>> len(t.children('XIX', type='group'))
11350

# Count immediate children
>>> len(t.children('S62', type=0))
9

# Count siblings
>>> len(t.siblings('S62'))
9

# Count all descendants
>>> len(t.children('S62'))
2229
```

## Hierarchy Types

The ICD-10 hierarchy consists of:

1. **Chapter** - Roman numerals (I-XXII)
2. **Range** - Letter-number ranges (A00-A09)
3. **Category** - 3-character codes (A00)
4. **Block** - 4-character codes (A00.0)
5. **Subblock** - 5-character codes (A00.00)
6. **Group** - 6-character codes (A00.001)
7. **Subgroup** - 7+ character codes (A00.001A)

## Examples

### Finding Information About Cholera

```python
# Multiple ways to get the same information
>>> t.get('A00')
>>> t.get('cholera')

# Get the code for cholera
>>> t.shift('cholera')
'A00'

# Get all cholera-related codes
>>> t.children('cholera')
['A00.0', 'A00.1', 'A00.9']

# Get cholera with names
>>> t.children('cholera', format='both')
['A00.0 Cholera due to Vibrio cholerae 01, biovar cholerae',
 'A00.1 Cholera due to Vibrio cholerae 01, biovar eltor',
 'A00.9 Cholera, unspecified']

# Get the full hierarchy
>>> t.hierarchy('cholera')
```

### Analyzing Fractures

```python
# Search for all fracture codes
>>> fractures = t.match('fracture')
>>> len(fractures)
20409

# Search only for fracture categories (main level)
>>> fracture_categories = t.match('fracture', type='category', format='both')
>>> fracture_categories[:3]
['S02 Fracture of skull and facial bones',
 'S12 Fracture of cervical vertebra and other parts of neck', 
 'S22 Fracture of rib(s), sternum and thoracic spine']

# Search only for fracture blocks (more specific)
>>> fracture_blocks = t.match('fracture', type='block')[:5]
['S02.0', 'S02.1', 'S02.2', 'S02.3', 'S02.4']

# Get specific fracture info
>>> t.get('S62')
{'code': 'S62', 'name': 'Fracture at wrist and hand level', ...}

# Count fracture subcategories
>>> len(t.children('S62', type='block'))
9

# Get all siblings (other wrist/hand injuries)
>>> t.siblings('S62', format='both')[:3]
['S60 Superficial injury of wrist, hand and fingers',
 'S61 Open wound of wrist, hand and fingers',
 'S63 Dislocation and sprain of joints and ligaments...']
```

### Working with Diseases by Type

```python
# Find all chapters containing 'disease'
>>> disease_chapters = t.match('disease', type='chapter', format='both')
>>> disease_chapters[:3]
['I Certain infectious and parasitic diseases',
 'II Neoplasms', 
 'III Diseases of the blood and blood-forming organs...']

# Find disease-related ranges
>>> disease_ranges = t.match('disease', type='range', format='name')[:3]
['Intestinal infectious diseases', 
 'Diseases of the blood and blood-forming organs', 
 'Endocrine, nutritional and metabolic diseases']

# Find specific disease categories
>>> heart_diseases = t.match('heart disease', type='category')[:3]
['I25', 'I27', 'I42']
```

### Working with Chapters

```python
# Get all chapters with names
>>> chapters = t.chapters(format='both')

# Analyze a specific chapter
>>> t.get('XIX')
{'code': 'XIX', 'name': 'Injury, poisoning and certain other consequences...'}

# Count all groups in the injury chapter
>>> len(t.children('XIX', type='group'))
11350

# Get immediate children (ranges) with names
>>> t.children('XIX', type=0, format='name')[:3]
['Injuries to the head', 'Injuries to the neck', 'Injuries to the thorax']
```

## Design Philosophy

This API follows a minimalist, zen-like design:

- **Simple names**: `get`, `shift`, `type`, `match`, etc.
- **Universal input**: All methods accept both codes and names
- **Consistent format**: All list methods support format parameter
- **Smart defaults**: Sensible behavior without configuration
- **Use len()**: Standard Python for all counting operations
- **Fast operations**: Lazy loading and intelligent caching

## Performance

- Initial load: ~1-2 seconds
- Subsequent operations: Microseconds (cached)
- Memory efficient: Loads data only when needed
- Smart caching: Frequently used operations are cached 