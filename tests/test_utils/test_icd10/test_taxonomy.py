import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.icd10 import ICD10Taxonomy

# Global variable for the taxonomy instance
t = None

def test(func_call: str):
    """Execute and display a function call."""
    print(f"\n>>> {func_call}")
    try:
        result = eval(func_call)
        print(f"{result}")
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    global t  # Use the global variable
    
    print("=" * 70)
    print("ICD-10 Taxonomy - Complete API Test")
    print("=" * 70)
    
    # Initialize
    print("\n## Initialization")
    t = ICD10Taxonomy()
    
    # Basic lookups with codes and names
    print("\n## Basic Lookups (works with both codes and names)")
    test("t.get('S62')")
    test("t.get('cholera')")  # Using name
    test("t.get('XIX')")
    test("t.get('S60-S69')")
    
    # Full key method
    print("\n## Full Key Method")
    test("t.fullkey('S62')")
    test("t.fullkey('cholera')")
    test("t.fullkey('XIX')")
    
    # Shift method - convert between codes and names
    print("\n## Shift Method (code ↔ name conversion)")
    test("t.shift('S62')")  # Code to name
    test("t.shift('Cholera')")  # Name to code
    test("t.shift('Fracture at wrist and hand level')")  # Name to code
    
    # Type detection
    print("\n## Type Detection (works with codes and names)")
    test("t.type('XIX')")
    test("t.type('S60-S69')")
    test("t.type('S62')")
    test("t.type('cholera')")  # Using name
    test("t.type('S62.302A')")
    
    # Match functionality (renamed from search)
    print("\n## Match (case insensitive search)")
    test("t.match('CHOLERA', exact=True)")
    test("t.match('fracture')[:5]")  # First 5 results
    test("t.match('injury', format='name')[:3]")  # With name format
    test("t.match('injury', format='both')[:2]")  # With both format
    
    # Match with type filter
    print("\n## Match with Type Filter")
    test("t.match('injury', type='category')[:5]")  # Only categories
    test("t.match('fracture', type='block')[:3]")  # Only blocks
    test("t.match('fracture', type='category', format='both')[:3]")  # Categories with both format
    test("len(t.match('disease', type='chapter'))")  # Count chapters with 'disease'
    test("t.match('infectious', type='range', format='name')[:2]")  # Ranges with names
    
    # Children with new syntax
    print("\n## Children (flexible type parameter)")
    test("t.children('S62', type=0)")  # Immediate children (depth 0)
    test("t.children('S62', type='block')")  # All blocks under S62
    test("t.children('cholera')")  # All descendants (default)
    test("t.children('XIX', type=1)[:5]")  # Children at depth 1
    test("len(t.children('S62'))")  # Total descendants count
    
    # Siblings with format
    print("\n## Siblings (same type only)")
    test("t.siblings('S62')[:5]")  # Default format: code
    test("t.siblings('S62', format='name')[:3]")  # Name format
    test("t.siblings('A00', format='both')[:3]")  # Both format
    test("len(t.siblings('S62'))")  # Count of siblings
    
    # Parents
    print("\n## Parents")
    test("t.parent('S62')")  # Immediate parent
    test("t.parent('S62.3', format='name')")  # With name format
    test("t.parents('S62.302A')")  # All parents (bottom-up)
    test("t.parents('S62.302A', format='both')[:2]")  # With both format
    
    # Path
    print("\n## Path (top-down)")
    test("t.path('S62.3')")  # Default: codes
    test("t.path('S62.3', format='name')")
    test("t.path('S62.3', format='both')")
    
    # Hierarchy - complete analysis
    print("\n## Hierarchy (complete analysis)")
    h = test("t.hierarchy('S62.3')")
    if h:
        print(f"\nDetailed hierarchy for S62.3:")
        print(f"  Code: {h['code']}")
        print(f"  Name: {h['name']}")
        print(f"  Type: {h['type']}")
        print(f"  Full Key: {h['full_key']}")
        print(f"  Parents: {h['parents']}")
        print(f"  Children counts: {h['children']}")
        print(f"  Immediate children: {h['n_children']}")
        print(f"  Siblings count: {h['n_siblings']}")
        print(f"  Path: {h['path']}")
    
    # Type-specific list methods
    print("\n## Type-Specific List Methods")
    test("len(t.chapters())")  # Count all chapters
    test("t.chapters()[:5]")  # First 5 chapter codes
    test("t.chapters(format='name')[:3]")  # Chapter names
    test("t.ranges()[:5]")  # First 5 ranges
    test("t.categories()[:5]")  # First 5 categories
    test("len(t.blocks())")  # Count all blocks
    test("len(t.groups())")  # Count all groups
    
    # Using len() for counts
    print("\n## Using len() for Counts")
    test("len(t.children('XIX', type='group'))")  # Groups in chapter XIX
    test("len(t.children('S60-S69', type='category'))")  # Categories in range
    test("len(t.children('S62', type=0))")  # Immediate children of S62
    
    print("\n" + "=" * 70)
    print("API Summary:")
    print("  • All methods work with both codes AND names")
    print("  • Case insensitive throughout")
    print("  • Format parameter: 'code', 'name', or 'both'")
    print("  • Match type filter: filter search by hierarchy type")
    print("  • Children type: hierarchy name, depth number, or None for all")
    print("  • Use len() for counting")
    print("  • Minimal, elegant, zen-like design")
    print("=" * 70)

if __name__ == "__main__":
    main() 