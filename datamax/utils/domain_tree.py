from typing import List, Dict, Optional, Any
import json

class DomainTree:
    """
    A tree structure utility class for caching and managing hierarchical labels.
    Supports node addition, deletion, update, search, path finding, and tree serialization......
    """
    def __init__(self, tree_data: Optional[List[Dict[str, Any]]] = None) -> None:
        self.tree: List[Dict[str, Any]] = tree_data or []
        
    #add node
    def add_node(self, label: str, parent_label: Optional[str] = None) -> bool:
        if parent_label is None:
            self.tree.append({"label": label})
            return True
        parent = self.find_node(parent_label)
        if parent is not None:
            if "child" not in parent:
                parent["child"] = []
            parent["child"].append({"label": label})
            return True
        return False

    #remove node
    def remove_node(self, label: str) -> bool:
        for index, node in enumerate(self.tree):
            if node.get("label") == label:
                del self.tree[index]
                return True
        for node in self.tree:
            if "child" in node:
                for child_index, child in enumerate(node["child"]):
                    if child.get("label") == label:
                        del node["child"][child_index]
                        return True
        return False

    #update node
    def update_node(self, old_label: str, new_label: str) -> bool:
        node = self.find_node(old_label)
        if node:
            node["label"] = new_label
            return True
        return False

    #find node
    def find_node(self, label: str) -> Optional[Dict[str, Any]]:
        for node in self.tree:
            if node.get("label") == label:
                return node
            if "child" in node:
                for child in node["child"]:
                    if child.get("label") == label:
                        return child
        return None

    #find path
    def find_path(self, label: str) -> Optional[str]:
        #recursive method need more parameters
        def _find(tree: List[Dict[str, Any]], label: str, path: List[str]) -> Optional[List[str]]:
            for node in tree:
                current_path = path + [node.get("label", "")]
                if node.get("label") == label:
                    return current_path
                if "child" in node:
                    result = _find(node["child"], label, current_path)
                    if result:
                        return result
            return None
        result = _find(self.tree, label, [])
        return "/".join(result) if result else None

        
    def to_json(self) -> List[Dict[str, Any]]:
        return self.tree

    def from_json(self, json_data: List[Dict[str, Any]]) -> None:
        self.tree = json_data

    def visualize(self) -> str:
        """
        visualization for domain tree
        
        Returns:string of tree structure
        """
        def _visualize_node(node: Dict[str, Any], level: int = 0) -> str:
            result = "  " * level + "├── " + node.get("label", "")
            if "child" in node and node["child"]:
                for i, child in enumerate(node["child"]):
                    if i == len(node["child"]) - 1:
                        result += "\n" + "  " * (level + 1) + "└── " + child.get("label", "")
                    else:
                        result += "\n" + _visualize_node(child, level + 1)
            return result
        
        if not self.tree:
            return "空树"
        
        result = "领域树结构:\n"
        for i, node in enumerate(self.tree):
            if i == len(self.tree) - 1:
                result += "└── " + node.get("label", "")
            else:
                result += _visualize_node(node, 0)
            result += "\n"
        
        return result

    def to_json_string(self) -> str:
        return json.dumps(self.tree, ensure_ascii=False, indent=2)

    def insert_node_between(self, node_name: str, parent_name: str, child_name: str) -> bool:
        """
        insert a new node between parent and child node
        
        :param node_name: name of the new node
        :param parent_name: name of the parent node new node will be inserted after
        :param child_name: name of the child node new node will be inserted before
        :return: whether the node is inserted successfully
        """
        # find parent node
        parent_node = self.find_node(parent_name)
        if not parent_node:
            return False
            
        # find child node
        child_node = self.find_node(child_name)
        if not child_node:
            return False
            
        # check if child node is really a direct child of parent node
        if "child" in parent_node:
            for i, child in enumerate(parent_node["child"]):
                if child.get("label") == child_name:
                    # create new node
                    new_node = {"label": node_name, "child": [child]}
                    # replace original child node
                    parent_node["child"][i] = new_node
                    return True
                    
        return False
if __name__ == "__main__":
    import json
    print("[DomainTree Main Interface Demo]")
    
    # 1. Create empty tree
    tree = DomainTree()
    print("***1. Empty tree: Create an empty tree structure.***", tree.to_json())

    # 2. Add root nodes
    tree.add_node("Computer Science")
    tree.add_node("Mathematics")
    print("***2. Add root nodes: Add root nodes 'Computer Science' and 'Mathematics'.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 3. Add child nodes
    tree.add_node("Artificial Intelligence", "Computer Science")
    tree.add_node("Machine Learning", "Artificial Intelligence")
    tree.add_node("Deep Learning", "Machine Learning")
    print("***3. Add child nodes: Add child nodes under the corresponding parent nodes.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 4. Find node
    node = tree.find_node("Machine Learning")
    print("***4. Find node: Find node 'Machine Learning', if not exist return None.***", node)

    # 5. Find path
    path = tree.find_path("Deep Learning")
    print("***5. Find path: Find the path of node 'Deep Learning', return path string or None.***", path)

    # 6. Update node
    tree.update_node("Machine Learning", "ML")
    print("***6. Update node: Update node label from 'Machine Learning' to 'ML'.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 7. Remove node
    tree.remove_node("ML")
    print("***7. Remove node: Remove node 'ML', if not exist do nothing.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 8. Export as JSON
    print("***8. Export as JSON: Export the tree structure as a JSON-serializable list.***", json.dumps(tree.to_json(), ensure_ascii=False, indent=2))

    # 9. Import from JSON
    json_data = [
        {"label": "Technology", "child": [
            {"label": "Programming Language", "child": [
                {"label": "Python"}, {"label": "Java"}
            ]}
        ]}
    ]
    tree.from_json(json_data)
    print("9. Import from JSON:", json.dumps(tree.to_json(), ensure_ascii=False, indent=2)) 