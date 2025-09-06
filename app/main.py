"""Main configuration editor application."""
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree, Label, Input, Button, Static, Select
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from .config_handler import ConfigHandler
from .theme import MONOKAI_THEME
from typing import Dict, Any


class ConfigEditorApp(App):
    """Textual app to edit config.yaml file with split layout and Monokai theme."""
    
    CSS = MONOKAI_THEME

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()
        self.config_handler = ConfigHandler(config_path)
        self.current_path = []
        self.current_value = None

    def compose(self) -> ComposeResult:
        """Create the UI layout."""
        yield Header()
        
        yield Container(
            ScrollableContainer(
                Tree("Configuration", id="config_tree"),
                id="tree_container"
            ),
            ScrollableContainer(
                Vertical(id="edit_panel", classes="edit-panel"),
                id="edit_container"
            )
        )
        
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the tree with config data."""
        tree = self.query_one("#config_tree", Tree)
        self.populate_tree(tree.root, self.config_handler.config, [])

    def populate_tree(self, node, data, path):
        """Recursively populate the tree with config data."""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = path + [key]
                child_node = node.add(key, data={"path": current_path, "value": value})
                
                if isinstance(value, dict):
                    self.populate_tree(child_node, value, current_path)
                elif isinstance(value, list):
                    # Handle lists by showing indices
                    for i, item in enumerate(value):
                        list_path = current_path + [i]
                        list_child = child_node.add(f"[{i}]", data={"path": list_path, "value": item})
                        if isinstance(item, dict):
                            self.populate_tree(list_child, item, list_path)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = path + [i]
                child_node = node.add(f"[{i}]", data={"path": current_path, "value": item})
                if isinstance(item, dict):
                    self.populate_tree(child_node, item, current_path)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle tree node selection."""
        if event.node.data:
            path = event.node.data["path"]
            value = event.node.data["value"]
            
            # Only show editor for leaf nodes (non-dict, non-list values)
            if not isinstance(value, (dict, list)):
                self.current_path = path
                self.current_value = value
                self.update_edit_panel()
            else:
                # Clear the edit panel for non-leaf nodes
                edit_panel = self.query_one("#edit_panel")
                edit_panel.remove_children()
                edit_panel.mount(
                    Vertical(
                        Label(f"Section: {' → '.join(map(str, path))}", classes="section"),
                        Label("Select a leaf node to edit its value.", classes="info")
                    )
                )

    def update_edit_panel(self):
        """Update the right panel with input for the selected parameter."""
        edit_panel = self.query_one("#edit_panel")
        edit_panel.remove_children()

        if self.current_path and self.current_value is not None:
            path_str = " → ".join(map(str, self.current_path))
            param_name = self.current_path[-1] if self.current_path else "Unknown"
            
            # Create appropriate input widget based on the current value type
            if isinstance(self.current_value, bool):
                # Use Select widget for boolean values
                input_widget = Select(
                    options=[("true", True), ("false", False)],
                    value=self.current_value,
                    allow_blank=False,
                    id="param_input"
                )
                type_info = "Boolean (true/false)"
            elif isinstance(self.current_value, int):
                input_widget = Input(
                    value=str(self.current_value),
                    placeholder="Integer value",
                    id="param_input"
                )
                type_info = "Integer"
            elif isinstance(self.current_value, float):
                input_widget = Input(
                    value=str(self.current_value),
                    placeholder="Float value",
                    id="param_input"
                )
                type_info = "Float"
            else:
                input_widget = Input(
                    value=str(self.current_value),
                    placeholder="Text value",
                    id="param_input"
                )
                type_info = "Text"
            
            edit_panel.mount(
                Vertical(
                    Label(f"Editing: {path_str}", classes="section"),
                    Label(f"Parameter: {param_name}"),
                    Label(f"Type: {type_info}"),
                    Label(f"Current value: {self.current_value}"),
                    input_widget,
                    Horizontal(
                        Button("Save", variant="success", id="save"),
                        Button("Reset", id="reset"),
                        Button("Exit", variant="error", id="exit")
                    )
                )
            )

    def on_button_pressed(self, event) -> None:
        """Handle button press events."""
        if event.button.id == "save":
            try:
                self.save_current_parameter()
                self.notify("Configuration saved successfully!", title="Success")
            except ValueError as e:
                self.notify(f"Error: Invalid input value - {str(e)}", title="Error", severity="error")
            except Exception as e:
                self.notify(f"Error: {str(e)}", title="Error", severity="error")
        
        elif event.button.id == "reset":
            # Reset the input to the original value
            if self.current_value is not None:
                input_widget = self.query_one("#param_input")
                if isinstance(input_widget, Select):
                    input_widget.value = self.current_value
                else:
                    input_widget.value = str(self.current_value)
        
        elif event.button.id == "exit":
            self.exit()

    def save_current_parameter(self):
        """Save the current parameter's input value to config."""
        if not self.current_path:
            return
        
        input_widget = self.query_one("#param_input")
        
        # Handle different widget types
        if isinstance(input_widget, Select):
            # For Select widgets (boolean values)
            new_value = input_widget.value
            # Validate that we have a proper boolean value
            if new_value is None or not isinstance(new_value, bool):
                raise ValueError("Please select either 'true' or 'false'")
        else:
            # For Input widgets (text, int, float)
            input_value = input_widget.value.strip()
            
            if not input_value and not isinstance(self.current_value, bool):
                return
            
            # Convert the input value to the appropriate type
            original_type = type(self.current_value)
            
            try:
                if original_type == bool:
                    new_value = input_value.lower() in ('true', 'yes', '1', 'on')
                elif original_type == int:
                    new_value = int(input_value)
                elif original_type == float:
                    new_value = float(input_value)
                else:
                    new_value = input_value
            except ValueError:
                raise ValueError(f"Cannot convert '{input_value}' to {original_type.__name__}")
        
        # Update the config using surgical approach
        success = self.config_handler.update_parameter_surgically(self.current_path, new_value)
        
        # Update the current value and refresh the tree while preserving expansion state
        self.current_value = new_value
        self.refresh_tree_preserving_state()

    def get_expanded_paths(self, node, current_path=[]):
        """Recursively collect all expanded node paths."""
        expanded_paths = []
        
        if hasattr(node, 'is_expanded') and node.is_expanded and node.data:
            expanded_paths.append(node.data["path"])
        
        if hasattr(node, 'children'):
            for child in node.children:
                expanded_paths.extend(self.get_expanded_paths(child, current_path))
        
        return expanded_paths

    def expand_nodes_by_paths(self, node, target_paths):
        """Recursively expand nodes based on their paths."""
        if node.data and node.data["path"] in target_paths:
            node.expand()
        
        if hasattr(node, 'children'):
            for child in node.children:
                self.expand_nodes_by_paths(child, target_paths)

    def refresh_tree_preserving_state(self):
        """Refresh the tree while preserving the expansion state."""
        tree = self.query_one("#config_tree", Tree)
        
        # Get currently expanded paths
        expanded_paths = self.get_expanded_paths(tree.root)
        
        # Get the currently selected node path
        selected_path = self.current_path if self.current_path else None
        
        # Clear and repopulate the tree
        tree.clear()
        tree.root.expand()
        self.populate_tree(tree.root, self.config_handler.config, [])
        
        # Restore expanded state
        self.expand_nodes_by_paths(tree.root, expanded_paths)
        
        # Try to restore selection if possible
        if selected_path:
            self.select_node_by_path(tree.root, selected_path)

    def select_node_by_path(self, node, target_path):
        """Recursively find and select a node by its path."""
        if node.data and node.data["path"] == target_path:
            tree = self.query_one("#config_tree", Tree)
            tree.select_node(node)
            return True
        
        if hasattr(node, 'children'):
            for child in node.children:
                if self.select_node_by_path(child, target_path):
                    return True
        
        return False


def main():
    """Entry point for the application."""
    app = ConfigEditorApp()
    app.run()


if __name__ == "__main__":
    main()
