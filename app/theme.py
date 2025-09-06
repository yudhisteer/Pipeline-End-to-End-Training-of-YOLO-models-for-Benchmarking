"""Monokai theme configuration for the app."""

MONOKAI_THEME = """
/* Monokai-inspired color scheme */
Container {
    layout: horizontal;
    height: 100%;
    background: #272822;
}

#tree_container {
    width: 40%;
    border: heavy #75715e;
    margin: 1;
    background: #272822;
}

#edit_container {
    width: 60%;
    border: heavy #75715e;
    margin: 1;
    background: #272822;
}

Tree {
    background: #272822;
    color: #f8f8f2;
    scrollbar-background: #49483e;
    scrollbar-color: #a6e22e;
}

Tree:focus {
    border: heavy #a6e22e;
}

TreeNode {
    background: #272822;
    color: #f8f8f2;
}

TreeNode:hover {
    background: #3e3d32;
}

TreeNode.-selected {
    background: #49483e;
    color: #a6e22e;
}

ScrollableContainer {
    scrollbar-background: #49483e;
    scrollbar-color: #a6e22e;
}

.edit-panel {
    padding: 1;
    background: #272822;
}

.info {
    color: #75715e;
    text-style: italic;
}

Input {
    margin: 0 1 1 1;
    background: #3e3d32;
    color: #f8f8f2;
    border: solid #75715e;
}

Input:focus {
    border: solid #a6e22e;
}

Select {
    margin: 0 1 1 1;
    background: #3e3d32;
    color: #f8f8f2;
    border: solid #75715e;
}

Select:focus {
    border: solid #a6e22e;
}

SelectOverlay {
    background: #3e3d32;
    color: #f8f8f2;
    border: solid #75715e;
}

OptionList {
    background: #3e3d32;
    color: #f8f8f2;
}

OptionList > .option-list--option {
    background: #3e3d32;
    color: #f8f8f2;
}

OptionList > .option-list--option:hover {
    background: #49483e;
}

OptionList > .option-list--option-selected {
    background: #a6e22e;
    color: #272822;
}

Button {
    margin: 1 1 0 1;
    background: #49483e;
    color: #f8f8f2;
    border: solid #75715e;
}

Button:hover {
    background: #75715e;
}

Button.-success {
    background: #a6e22e;
    color: #272822;
}

Button.-error {
    background: #f92672;
    color: #272822;
}

Label {
    margin: 0 1 0 1;
    color: #f8f8f2;
}

.section {
    padding: 1;
    color: #a6e22e;
    text-style: bold;
}

Header {
    background: #272822;
    color: #f8f8f2;
}

Footer {
    background: #272822;
    color: #f8f8f2;
}

Static {
    background: #272822;
    color: #f8f8f2;
}

Vertical {
    background: #272822;
}

Horizontal {
    background: #272822;
}
"""
