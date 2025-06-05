import json
import os
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict, Counter

class ICD10Taxonomy:
    """
    Minimalist ICD-10 Taxonomy with zen-like API.
    Fast, elegant, and comprehensive hierarchical analysis.
    """
    
    def __init__(self, data_file_name: str = "icd10-taxonomy-complete.json") -> None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.json_file_path = os.path.join(data_dir, data_file_name)
        
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"Data file '{self.json_file_path}' not found.")
        
        # Core data cache
        self._raw_data: Optional[Dict[str, Any]] = None
        self._code_map: Dict[str, Dict] = {}  # Maps codes to full data
        self._name_map: Dict[str, str] = {}   # Maps names to codes
        self._hierarchy_built = False
        
        # Caches
        self._cache = {}
        
        print(f"ICD10Taxonomy initialized: {self.json_file_path}")
    
    def _load_data(self) -> Dict[str, Any]:
        """Load JSON data lazily."""
        if self._raw_data is None:
            print("Loading ICD-10 data...")
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                self._raw_data = json.load(f)
            print("Data loaded.")
            self._build_maps()
        return self._raw_data
    
    def _build_maps(self) -> None:
        """Build code and name maps."""
        if self._hierarchy_built:
            return
            
        self._code_map = {}
        self._name_map = {}
        self._build_maps_recursive(self._raw_data, [])
        self._hierarchy_built = True
    
    def _build_maps_recursive(self, data: dict, path: List[str]) -> None:
        """Recursively build the maps."""
        for key, value in data.items():
            if key == "..." or (isinstance(value, str) and value == "..."):
                continue
                
            current_path = path + [key]
            
            # Extract simplified code from key
            simple_code = self._extract_code(key)
            name = self._extract_name(key, value)
            
            # Store by simple code if available
            if simple_code:
                self._code_map[simple_code] = {
                    'code': simple_code,
                    'full_key': key,
                    'name': name,
                    'data': value,
                    'path': current_path,
                    'parent_path': path,
                    'type': self._get_type(key)
                }
                # Map name to code (case insensitive)
                if name:
                    self._name_map[name.lower()] = simple_code
            
            # Also store the full key
            self._code_map[key] = {
                'code': key,
                'full_key': key,
                'name': name,
                'data': value,
                'path': current_path,
                'parent_path': path,
                'type': self._get_type(key)
            }
            # Map name for full key too
            if name and key not in self._name_map:
                self._name_map[name.lower()] = key
            
            if isinstance(value, dict):
                # Handle sub_codes
                if "sub_codes" in value:
                    for sub_item in value.get("sub_codes", []):
                        if isinstance(sub_item, dict) and "code" in sub_item:
                            sub_code = sub_item["code"]
                            sub_name = sub_item.get("name", "")
                            sub_path = current_path + [sub_code]
                            
                            self._code_map[sub_code] = {
                                'code': sub_code,
                                'full_key': sub_code,
                                'name': sub_name,
                                'data': sub_item,
                                'path': sub_path,
                                'parent_path': current_path,
                                'type': self._get_type(sub_code)
                            }
                            
                            if sub_name:
                                self._name_map[sub_name.lower()] = sub_code
                
                # Recursive
                self._build_maps_recursive(value, current_path)
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract the code part from a full key."""
        # Roman numerals at start
        match = re.match(r'^([IVXLCDM]+)\s', text)
        if match:
            return match.group(1)
        
        # Range codes
        match = re.match(r'^([A-Z]\d{2}-[A-Z]\d{2})\s', text)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_name(self, key: str, value: Any) -> str:
        """Extract proper name from key/value."""
        if isinstance(value, dict) and 'name' in value:
            return value['name']
        
        # For chapters/ranges, extract the descriptive part
        # Roman numerals
        match = re.match(r'^[IVXLCDM]+\s+(.+)$', key)
        if match:
            return match.group(1)
        
        # Range codes
        match = re.match(r'^[A-Z]\d{2}-[A-Z]\d{2}\s+(.+)$', key)
        if match:
            return match.group(1)
        
        # If no pattern matches, return the key itself
        return key
    
    def _resolve_input(self, code_or_name: str) -> Optional[str]:
        """Resolve input to a code, whether it's a code or name."""
        self._load_data()
        
        # Try as code first (case insensitive)
        upper = code_or_name.upper()
        if upper in self._code_map:
            return upper
        
        # Try as name (case insensitive)
        lower = code_or_name.lower()
        if lower in self._name_map:
            return self._name_map[lower]
        
        return None
    
    def _get_type(self, code: str) -> str:
        """Determine hierarchy type."""
        if not code: 
            return "unknown"
            
        # Chapter (Roman numerals)
        if re.match(r'^[IVXLCDM]+(\s|$)', code):
            return "chapter"
            
        # Range
        if re.match(r'^[A-Z]\d{2}-[A-Z]\d{2}', code):
            return "range"
        
        # Standard codes
        base = re.match(r'^[A-Z]\d{2}', code)
        if not base: 
            return "unknown"
        
        ext = code[len(base.group(0)):]
        
        if not ext: 
            return "category"
        if ext.startswith('.'):
            ext_part = ext[1:]
            if len(ext_part) >= 4 and re.match(r'\d{1,3}[A-Z0-9]$', ext_part): 
                return "subgroup"
            if re.match(r'\d{3}$', ext_part): 
                return "group"
            if re.match(r'\d{2}$', ext_part): 
                return "subblock"
            if re.match(r'\d{1}$', ext_part): 
                return "block"
        return "unknown"
    
    def _format_result(self, code: str, format: str = "code") -> str:
        """Format a single result according to format parameter."""
        if format == "code":
            return code
        elif format == "name":
            info = self._code_map.get(code, {})
            return info.get('name', code)
        elif format == "both":
            info = self._code_map.get(code, {})
            name = info.get('name', '')
            return f"{code} {name}" if name else code
        return code
    
    def _format_list(self, codes: List[str], format: str = "code") -> List[str]:
        """Format a list of codes according to format parameter."""
        return [self._format_result(code, format) for code in codes]
    
    # ========== CORE API - Minimalist & Elegant ==========
    
    def get(self, code_or_name: str) -> Optional[Dict]:
        """Get basic info for a code or name."""
        code = self._resolve_input(code_or_name)
        if not code:
            return None
            
        if code in self._code_map:
            info = self._code_map[code]
            return {
                'code': info['code'],
                'name': info['name'],
                'type': info['type'],
                'full_key': f"{info['code']} {info['name']}" if info['name'] else info['code']
            }
        return None
    
    def fullkey(self, code_or_name: str) -> Optional[str]:
        """Get full key (code + name) for any input."""
        info = self.get(code_or_name)
        return info['full_key'] if info else None
    
    def shift(self, code_or_name: str) -> Optional[str]:
        """Convert code to name or name to code."""
        self._load_data()
        
        # If it looks like a code, return the name
        upper = code_or_name.upper()
        if upper in self._code_map:
            return self._code_map[upper]['name']
        
        # If it's a name, return the code
        lower = code_or_name.lower()
        if lower in self._name_map:
            return self._name_map[lower]
        
        return None
    
    def type(self, code_or_name: str) -> Optional[str]:
        """Get hierarchy type of a code or name."""
        code = self._resolve_input(code_or_name)
        if not code:
            return None
            
        if code in self._code_map:
            return self._code_map[code]['type']
        return None
    
    def match(self, query: str, exact: bool = False, type: Optional[str] = None, format: str = "code") -> List[str]:
        """Search by name, returns list. Case insensitive. Optional type filter."""
        self._load_data()
        query = query.lower()
        results = []
        
        for code, info in self._code_map.items():
            name = info['name'].lower() if info['name'] else ""
            
            # Apply type filter if specified
            if type and info['type'] != type:
                continue
            
            if exact:
                if name == query:
                    results.append(code)
            else:
                if query in name:
                    results.append(code)
        
        # Filter to main codes only (not sub-codes) if exact match
        if exact and results:
            filtered = []
            for code in results:
                # Skip if it's a sub_code and we have its parent
                parent_code = self.parent(code)
                is_subcode = parent_code in results if parent_code else False
                if not is_subcode:
                    filtered.append(code)
            results = filtered
        
        return self._format_list(results, format)
    
    def children(self, code_or_name: str, type: Optional[Union[str, int]] = None, format: str = "code") -> List[str]:
        """Get children of a code. Type can be hierarchy name or depth (0=immediate, 1=next level, etc)."""
        code = self._resolve_input(code_or_name)
        if not code:
            return []
        
        if type is None:
            # Get all descendants
            all_children = []
            self._collect_all_children(code, all_children)
            return self._format_list(all_children, format)
        
        elif isinstance(type, int):
            # Get children at specific depth
            return self._format_list(self._get_children_at_depth(code, type), format)
        
        else:
            # Get children of specific type
            children = []
            self._collect_children_of_type(code, type, children)
            return self._format_list(children, format)
    
    def _get_children_at_depth(self, code: str, depth: int) -> List[str]:
        """Get children at specific depth level."""
        if depth == 0:
            return self._immediate_children(code)
        
        current_level = [code]
        for _ in range(depth + 1):
            next_level = []
            for c in current_level:
                next_level.extend(self._immediate_children(c))
            current_level = next_level
        
        return current_level
    
    def _collect_children_of_type(self, code: str, target_type: str, result: List[str]) -> None:
        """Recursively collect children of specific type."""
        for child in self._immediate_children(code):
            if self.type(child) == target_type:
                result.append(child)
            self._collect_children_of_type(child, target_type, result)
    
    def _immediate_children(self, code: str) -> List[str]:
        """Get immediate children only."""
        children = []
        
        if code in self._code_map:
            info = self._code_map[code]
            data = info['data']
            
            if isinstance(data, dict):
                # Check sub_codes
                if "sub_codes" in data:
                    for sub in data["sub_codes"]:
                        if isinstance(sub, dict) and "code" in sub:
                            children.append(sub["code"])
                
                # Check nested codes
                for key, value in data.items():
                    if key not in ["name", "sub_codes"] and isinstance(value, dict):
                        # Extract simple code if available
                        simple = self._extract_code(key)
                        children.append(simple if simple else key)
        
        return children
    
    def _collect_all_children(self, code: str, result: List[str]) -> None:
        """Recursively collect all children."""
        for child in self._immediate_children(code):
            result.append(child)
            self._collect_all_children(child, result)
    
    def siblings(self, code_or_name: str, format: str = "code") -> List[str]:
        """Get siblings of same type at same level."""
        code = self._resolve_input(code_or_name)
        if not code:
            return []
        
        my_type = self.type(code)
        parent_code = self.parent(code)
        
        if parent_code:
            # Get all children of parent with same type
            all_siblings = self._immediate_children(parent_code)
            siblings = [s for s in all_siblings if s != code and self.type(s) == my_type]
        else:
            # Root level - get all roots of same type
            siblings = [c for c, info in self._code_map.items() 
                       if not info['parent_path'] and info['type'] == my_type and c != code]
        
        return self._format_list(siblings, format)
    
    def parent(self, code_or_name: str, format: str = "code") -> Optional[str]:
        """Get immediate parent."""
        code = self._resolve_input(code_or_name)
        if not code:
            return None
        
        if code in self._code_map:
            parent_path = self._code_map[code]['parent_path']
            if parent_path:
                parent_key = parent_path[-1]
                # Return simplified code if available
                simple = self._extract_code(parent_key)
                parent_code = simple if simple else parent_key
                return self._format_result(parent_code, format)
        return None
    
    def parents(self, code_or_name: str, format: str = "code") -> List[str]:
        """Get all parents up to root (opposite of path, bottom-up)."""
        parents = []
        current = code_or_name
        
        while True:
            parent_code = self.parent(current, format="code")  # Always get code for iteration
            if not parent_code:
                break
            parents.append(parent_code)
            current = parent_code
        
        return self._format_list(parents, format)
    
    def path(self, code_or_name: str, format: str = "code") -> List[str]:
        """Get full path from root to code (top-down)."""
        code = self._resolve_input(code_or_name)
        if not code or code not in self._code_map:
            return []
        
        path_keys = self._code_map[code]['path']
        codes = []
        
        for key in path_keys:
            # Extract code or use key
            simple = self._extract_code(key)
            codes.append(simple if simple else key)
        
        return self._format_list(codes, format)
    
    def hierarchy(self, code_or_name: str) -> Dict[str, Any]:
        """Get complete hierarchy info in one call."""
        code = self._resolve_input(code_or_name)
        if not code:
            return {}
        
        info = self.get(code)
        if not info:
            return {}
        
        # Build hierarchy
        result = {
            'code': info['code'],
            'name': info['name'],
            'type': info['type'],
            'full_key': info['full_key'],
            'parents': {},
            'children': {},
            'n_children': len(self.children(code, type=0)),  # Immediate children
            'n_siblings': len(self.siblings(code)),
            'path': self.path(code)
        }
        
        # Parents by type
        for parent_code in self.parents(code):
            p_info = self.get(parent_code)
            if p_info:
                result['parents'][p_info['type']] = {
                    'code': parent_code,
                    'name': p_info['name']
                }
        
        # Children counts by type
        types = ["chapter", "range", "category", "block", "subblock", "group", "subgroup"]
        for t in types:
            children_of_type = self.children(code, type=t)
            if children_of_type:
                result['children'][t] = len(children_of_type)
        
        return result
    
    # ========== Type-specific list methods ==========
    
    def chapters(self, format: str = "code") -> List[str]:
        """Get all chapters."""
        self._load_data()
        chapters = [code for code, info in self._code_map.items() if info['type'] == 'chapter']
        return self._format_list(chapters, format)
    
    def ranges(self, format: str = "code") -> List[str]:
        """Get all ranges."""
        self._load_data()
        ranges = [code for code, info in self._code_map.items() if info['type'] == 'range']
        return self._format_list(ranges, format)
    
    def categories(self, format: str = "code") -> List[str]:
        """Get all categories."""
        self._load_data()
        categories = [code for code, info in self._code_map.items() if info['type'] == 'category']
        return self._format_list(categories, format)
    
    def blocks(self, format: str = "code") -> List[str]:
        """Get all blocks."""
        self._load_data()
        blocks = [code for code, info in self._code_map.items() if info['type'] == 'block']
        return self._format_list(blocks, format)
    
    def subblocks(self, format: str = "code") -> List[str]:
        """Get all subblocks."""
        self._load_data()
        subblocks = [code for code, info in self._code_map.items() if info['type'] == 'subblock']
        return self._format_list(subblocks, format)
    
    def groups(self, format: str = "code") -> List[str]:
        """Get all groups."""
        self._load_data()
        groups = [code for code, info in self._code_map.items() if info['type'] == 'group']
        return self._format_list(groups, format)
    
    def subgroups(self, format: str = "code") -> List[str]:
        """Get all subgroups."""
        self._load_data()
        subgroups = [code for code, info in self._code_map.items() if info['type'] == 'subgroup']
        return self._format_list(subgroups, format)