"""Main configuration editor application."""
from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Tree, Label, Input, Button, Static
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
            
            # Determine the input type based on the current value
            if isinstance(self.current_value, bool):
                placeholder = "true/false"
            elif isinstance(self.current_value, int):
                placeholder = "Integer value"
            elif isinstance(self.current_value, float):
                placeholder = "Float value"
            else:
                placeholder = "Text value"
            
            edit_panel.mount(
                Vertical(
                    Label(f"Editing: {path_str}", classes="section"),
                    Label(f"Parameter: {param_name}"),
                    Label(f"Current value: {self.current_value}"),
                    Input(
                        value=str(self.current_value),
                        placeholder=placeholder,
                        id="param_input"
                    ),
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
                input_widget.value = str(self.current_value)
        
        elif event.button.id == "exit":
            self.exit()

    def save_current_parameter(self):
        """Save the current parameter's input value to config."""
        if not self.current_path:
            return
        
        input_widget = self.query_one("#param_input")
        input_value = input_widget.value.strip()
        
        if not input_value:
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
        
        # Update the current value and refresh the tree
        self.current_value = new_value
        tree = self.query_one("#config_tree", Tree)
        tree.clear()
        tree.root.expand()
        self.populate_tree(tree.root, self.config_handler.config, [])



def main():
    """Entry point for the application."""
    app = ConfigEditorApp()
    app.run()


if __name__ == "__main__":
    main()
