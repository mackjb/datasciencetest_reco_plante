import streamlit.components.v1 as components
import json

def render_mermaid(mindmap_data, height=800):
    """
    Renders an interactive Mermaid mindmap.
    
    Args:
        mindmap_data (dict or str): The mindmap definition.
            - If dict: Treated as a tree structure for interactive rendering.
              Format: {"id": "root", "text": "Root", "children": [...], "collapsed": False}
            - If str: Treated as a raw Mermaid string (static rendering).
        height (int): Height of the component in pixels.
    """
    
    # Check if backward compatibility (string/static) is needed
    if isinstance(mindmap_data, str):
        mode = "static"
        data_json = json.dumps(mindmap_data)
    else:
        mode = "interactive"
        data_json = json.dumps(mindmap_data)

    html_code = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; overflow: hidden; font-family: sans-serif; }}
        #mermaid-container {{ 
            width: 100vw; 
            height: 100vh; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            cursor: move; /* Indicate pannable area */
        }}
        .mermaid {{ width: 100%; height: 100%; }}
        
        /* Interactive Node Styling */
        .interactive-node {{ 
            cursor: pointer !important; 
            pointer-events: all !important;
        }}
        .interactive-node text {{
            user-select: none; /* Prevent text selection on click */
        }}
        .interactive-node:hover rect, .interactive-node:hover circle, .interactive-node:hover polygon {{
            filter: brightness(0.95);
            stroke-width: 3px !important;
        }}
        
        /* Tooltip styling */
        #tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            z-index: 1000;
        }}

        /* Fix truncated content */
        svg {{
            max-width: none !important;
            height: 100% !important;
            width: 100% !important;
        }}

        /* Controls styling */
        .controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1000;
            display: flex;
            gap: 10px;
        }}
        .control-btn {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 5px 10px;
            cursor: pointer;
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s;
        }}
        .control-btn:hover {{
            background: #f0f0f0;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        }}
    </style>
</head>
<body>
    <div id="tooltip">Click to expand/collapse</div>
    
    <!-- Global Controls -->
    <div class="controls">
        <button class="control-btn" onclick="expandAll()">Expand All</button>
        <button class="control-btn" onclick="collapseAll()">Collapse All</button>
    </div>

    <div id="mermaid-container">
        <div class="mermaid" id="mermaid-graph"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>

    <script>
        const mode = "{mode}";
        let data = {data_json};
        let panZoomInstance = null;

        mermaid.initialize({{ 
            startOnLoad: false, 
            securityLevel: 'loose',
            theme: 'base',
            mindmap: {{
                useMaxWidth: false,
            }}
        }});

        // --- Helper: Generate Mermaid Syntax from Data Tree ---
        function generateMermaidTrace(node, depth=0) {{
            let indent = " ".repeat(depth * 2);
            // Default shape/id logic
            let nodeId = node.id || "node_" + Math.random().toString(36).substr(2, 9);
            // Escape text for Mermaid
            let label = node.text.replace(/"/g, "'");
            
            // Visual cue for children if collapsed
            if (node.collapsed && node.children && node.children.length > 0) {{
                label += " [ + ]";
            }}

            // Formatting based on depth or custom shape provided in data could be added here
            // Simple syntax: separate ID and Label? Mindmap syntax is loose.
            // Using standard indentation-based syntax:
            // Root
            //   Child
            
            // For root (depth 0), logic is slightly different in Mermaid mindmaps (start with 'mindmap\\n  root((Label))')
            
            let line = "";
            
            if (depth === 0) {{
                // Root node styling
                line = `${{indent}}root((<b>${{label}}</b>))`; 
            }} else {{
                // Child styling. Can check node type/shape if provided
                // Using generic brackets for clarity, or plain text
                // Adjust shape based on level?
                if (depth === 1) {{
                    line = `${{indent}}${{nodeId}}("${{label}}")`;
                }} else {{
                    line = `${{indent}}${{nodeId}}["${{label}}"]`; 
                }}
            }}

            let text = line + "\\n";
            
            if (!node.collapsed && node.children) {{
                node.children.forEach(child => {{
                    text += generateMermaidTrace(child, depth + 1);
                }});
            }}
            
            return text;
        }}

        function generateMermaid(data) {{
            if (mode === 'static') return data; // It's already a string
            
            let syntax = "mindmap\\n";
            syntax += generateMermaidTrace(data, 0);
            return syntax;
        }}

        // --- Core Rendering Function ---
        async function render() {{
            const graphDiv = document.getElementById('mermaid-graph');
            const syntax = generateMermaid(data);
            
            // Clean up previous SVG and panzoom to prevent leaks/errors
            if (panZoomInstance) {{
                panZoomInstance.destroy();
                panZoomInstance = null;
            }}
            graphDiv.innerHTML = '';
            
            try {{
                const {{ svg }} = await mermaid.render('mermaid-svg-' + Date.now(), syntax, graphDiv);
                graphDiv.innerHTML = svg;
                
                // Enhance SVG for full container
                const svgElement = graphDiv.querySelector('svg');
                svgElement.style.width = '100%';
                svgElement.style.height = '100%';
                svgElement.style.maxWidth = 'none';

                // Setup PanZoom
                panZoomInstance = svgPanZoom(svgElement, {{
                    zoomEnabled: true,
                    controlIconsEnabled: true,
                    fit: true,
                    center: true, 
                    minZoom: 0.1,
                    maxZoom: 10
                }});
                
                // Initial Zoom Step (Custom preference)
                panZoomInstance.zoomBy(1.2); 

                // Setup Click Interactivity
                if (mode === 'interactive') {{
                    setupInteractivity(svgElement);
                }}

            }} catch (err) {{
                console.error("Mermaid Render Error:", err);
                graphDiv.innerHTML = "<div style='color:red'>Error rendering mindmap</div>";
            }}
        }}

        // --- Interactivity Logic ---
        function setupInteractivity(svgElement) {{
            // Create a flat map of ID -> Node for easier lookup
            const nodeMap = {{}};
            function buildNodeMap(node) {{
                if (node.id) nodeMap[node.id] = node;
                if (node.children) node.children.forEach(buildNodeMap);
            }}
            buildNodeMap(data);

            const nodes = svgElement.querySelectorAll('g.mindmap-node');
            
            nodes.forEach(nodeGroup => {{
                // 1. Try matching by ID (Mermaid often assigns the ID we defined to the group or an inner element)
                let dataNode = null;
                
                // Check the group's ID or class
                // Mermaid might prefix ids? e.g. "flowchart-id-..."
                // But in mindmap, it often uses the ID directly or close to it.
                // We check if any mapped ID matches the group ID
                if (nodeGroup.id && nodeMap[nodeGroup.id]) {{
                    dataNode = nodeMap[nodeGroup.id];
                }}

                if (!dataNode) {{
                    // 2. Fallback to Text Matching
                    const textEl = nodeGroup.querySelector('text');
                    if (textEl) {{
                        const nodeText = textEl.textContent.trim().replace('[ + ]', '').trim();
                        dataNode = findNodeByText(data, nodeText);
                    }}
                }}
                
                if (dataNode && dataNode.children && dataNode.children.length > 0) {{
                    nodeGroup.classList.add('interactive-node');
                    
                    let downX, downY;
                    nodeGroup.addEventListener('mousedown', (e) => {{
                        downX = e.clientX;
                        downY = e.clientY;
                    }});
                    
                    nodeGroup.addEventListener('mouseup', (e) => {{
                        const diffX = Math.abs(e.clientX - downX);
                        const diffY = Math.abs(e.clientY - downY);
                        
                        if (diffX < 5 && diffY < 5) {{ 
                            e.stopPropagation(); 
                            toggleNode(dataNode);
                        }}
                    }});
                    
                    nodeGroup.addEventListener('touchstart', (e) => {{
                        const touch = e.touches[0];
                        downX = touch.clientX;
                        downY = touch.clientY;
                    }});
                     nodeGroup.addEventListener('touchend', (e) => {{
                        const touch = e.changedTouches[0];
                        const diffX = Math.abs(touch.clientX - downX);
                        const diffY = Math.abs(touch.clientY - downY);
                        if (diffX < 5 && diffY < 5) {{
                             toggleNode(dataNode);
                        }}
                    }});
                }}
            }});
        }}

        function findNodeByText(root, text) {{
            // Aggressive normalization: remove HTML, remove ALL whitespace, lowercase
            const clean = s => s.replace(/<[^>]*>/g, '').replace(/\s+/g, '').toLowerCase();
            const target = clean(text);
            const current = clean(root.text);
            
            // Exact match on cleaned text
            if (current === target) return root;
            
            // Substring match (robustness for partial rendering) usually unsafe, but with strict cleaning OK
            // Only if lengths are very close
            if ((current.includes(target) || target.includes(current)) && 
                Math.abs(current.length - target.length) < 5) {{
                return root;
            }}
            
            if (root.children) {{
                for (let child of root.children) {{
                    const found = findNodeByText(child, text);
                    if (found) return found;
                }}
            }}
            return null;
        }}

        function toggleNode(node) {{
            node.collapsed = !node.collapsed;
            render(); // Re-render with new state
        }}

        // --- Global Expand/Collapse ---
        function setAllCollapsed(node, state) {{
            if (node.children) {{
                node.collapsed = state;
                node.children.forEach(child => setAllCollapsed(child, state));
            }}
        }}

        function expandAll() {{
            setAllCollapsed(data, false);
            render();
        }}

        function collapseAll() {{
            // Keep root expanded usually, or just collapse everything children of root
            if (data.children) {{
                 data.children.forEach(child => setAllCollapsed(child, true));
            }}
            render();
        }}

        // Initial Render
        render();

        // Handle window resize
        window.addEventListener('resize', () => {{
            if (panZoomInstance) panZoomInstance.resize();
        }});
    </script>
</body>
</html>
    """
    components.html(html_code, height=height)

