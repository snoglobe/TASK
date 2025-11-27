#!/usr/bin/env python3
"""
TASK format trace viewer - web-based UI with syntax highlighting.
"""

import argparse
import json
import re
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# HTML template with syntax highlighting
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TASK Trace Viewer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        
        .container {
            display: flex;
            height: 100vh;
        }
        
        /* Sidebar */
        .sidebar {
            width: 280px;
            background: #12121a;
            border-right: 1px solid #2a2a3a;
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }
        
        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid #2a2a3a;
        }
        
        .sidebar-header h1 {
            font-size: 18px;
            font-weight: 600;
            color: #fff;
            margin-bottom: 8px;
        }
        
        .stats {
            font-size: 12px;
            color: #888;
        }
        
        .search {
            padding: 12px 20px;
            border-bottom: 1px solid #2a2a3a;
        }
        
        .search input {
            width: 100%;
            padding: 8px 12px;
            background: #1a1a24;
            border: 1px solid #2a2a3a;
            border-radius: 6px;
            color: #e0e0e0;
            font-size: 13px;
            outline: none;
        }
        
        .search input:focus {
            border-color: #4a9eff;
        }
        
        .trace-list {
            flex: 1;
            overflow-y: auto;
            padding: 8px 0;
        }
        
        .trace-item {
            padding: 10px 20px;
            cursor: pointer;
            border-left: 3px solid transparent;
            transition: all 0.15s;
        }
        
        .trace-item:hover {
            background: #1a1a24;
        }
        
        .trace-item.active {
            background: #1a1a2a;
            border-left-color: #4a9eff;
        }
        
        .trace-item .name {
            font-size: 13px;
            font-weight: 500;
            color: #fff;
        }
        
        .trace-item .meta {
            font-size: 11px;
            color: #666;
            margin-top: 2px;
        }
        
        .trace-item .tags {
            margin-top: 4px;
        }
        
        .tag {
            display: inline-block;
            padding: 2px 6px;
            font-size: 10px;
            border-radius: 4px;
            margin-right: 4px;
        }
        
        .tag.tools { background: #2a4a3a; color: #6ecf8e; }
        .tag.no-tools { background: #4a3a2a; color: #cfae6e; }
        .tag.multi-turn { background: #3a2a4a; color: #ae6ecf; }
        .tag.error { background: #4a2a2a; color: #cf6e6e; }
        
        /* Main content */
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .toolbar {
            padding: 12px 24px;
            background: #12121a;
            border-bottom: 1px solid #2a2a3a;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .toolbar .title {
            font-size: 14px;
            font-weight: 600;
            color: #fff;
        }
        
        .toolbar .info {
            font-size: 12px;
            color: #888;
        }
        
        .trace-content {
            flex: 1;
            overflow: auto;
            padding: 24px;
        }
        
        .trace-view {
            max-width: 900px;
            line-height: 1.6;
            font-size: 13px;
        }
        
        /* Syntax highlighting */
        .kw { color: #c678dd; font-weight: 500; }
        .op { color: #56b6c2; }
        .str { color: #98c379; }
        .special-str { color: #e5c07b; }
        .tag-marker { color: #e06c75; }
        .ref-marker { color: #61afef; }
        .satisfies { color: #d19a66; }
        .confidence { color: #c678dd; }
        .num { color: #d19a66; }
        .comment { color: #5c6370; font-style: italic; }
        
        .block {
            margin: 16px 0;
            padding: 16px;
            background: #12121a;
            border-radius: 8px;
            border: 1px solid #2a2a3a;
        }
        
        .block.system { border-left: 3px solid #61afef; }
        .block.tool { border-left: 3px solid #c678dd; }
        .block.user { border-left: 3px solid #98c379; }
        .block.plan { border-left: 3px solid #e5c07b; }
        .block.act { border-left: 3px solid #56b6c2; }
        .block.result { border-left: 3px solid #d19a66; }
        .block.response { border-left: 3px solid #e06c75; }
        
        .empty-state {
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #666;
            font-size: 14px;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #2a2a3a; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #3a3a4a; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>TASK Traces</h1>
                <div class="stats" id="stats">Loading...</div>
            </div>
            <div class="search">
                <input type="text" id="search" placeholder="Search traces..." />
            </div>
            <div class="trace-list" id="trace-list"></div>
        </div>
        <div class="main">
            <div class="toolbar">
                <span class="title" id="toolbar-title">Select a trace</span>
                <span class="info" id="toolbar-info"></span>
            </div>
            <div class="trace-content">
                <div class="trace-view" id="trace-view">
                    <div class="empty-state">Select a trace from the sidebar</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const traces = TRACES_DATA;
        let currentIndex = -1;
        
        // Syntax highlighting - use placeholders to avoid double-replacement
        function highlight(text) {
            const placeholders = [];
            let placeholderIndex = 0;
            
            function placeholder(content) {
                const key = `__PH${placeholderIndex++}__`;
                placeholders.push({ key, content });
                return key;
            }
            
            // Escape HTML first
            text = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            
            // Special quotes (capture first to protect content)
            text = text.replace(/「([^」]*)」/g, (_, content) => 
                placeholder(`<span class="special-str">「${content}」</span>`));
            
            // Regular strings
            text = text.replace(/"([^"]*)"/g, (_, content) => 
                placeholder(`<span class="str">"${content}"</span>`));
            
            // Comments
            text = text.replace(/(\/\/.*$)/gm, (_, content) => 
                placeholder(`<span class="comment">${content}</span>`));
            
            // Tags 🏷
            text = text.replace(/(🏷)\s*(\S+)/g, (_, marker, tag) => 
                placeholder(`<span class="tag-marker">${marker} ${tag}</span>`));
            
            // References ※
            text = text.replace(/(※)\s*(\[[^\]]+\]|\S+)/g, (_, marker, ref) => 
                placeholder(`<span class="ref-marker">${marker} ${ref}</span>`));
            
            // Satisfies ⊨
            text = text.replace(/(⊨)\s*(\d+)/g, (_, marker, num) => 
                placeholder(`<span class="satisfies">${marker} ${num}</span>`));
            
            // Confidence 𝑝
            text = text.replace(/(𝑝)\s*([\d.]+)/g, (_, marker, num) => 
                placeholder(`<span class="confidence">${marker} ${num}</span>`));
            
            // Operators (do before keywords to not break them)
            text = text.replace(/↦/g, () => placeholder('<span class="op">↦</span>'));
            text = text.replace(/•/g, () => placeholder('<span class="op">•</span>'));
            
            // Keywords (whole words only)
            text = text.replace(/\\b(system|tool|user|plan|act|result|response|todo|call|think|data|error|name|params|type|enum|required|description|rationale|id)\\b/g, 
                (_, kw) => placeholder(`<span class="kw">${kw}</span>`));
            
            // Restore placeholders (replace all occurrences)
            for (const { key, content } of placeholders) {
                text = text.split(key).join(content);
            }
            
            return text;
        }
        
        function getBlockType(text) {
            if (text.startsWith('system')) return 'system';
            if (text.startsWith('tool')) return 'tool';
            if (text.startsWith('user')) return 'user';
            if (text.startsWith('plan')) return 'plan';
            if (text.startsWith('act')) return 'act';
            if (text.startsWith('result')) return 'result';
            if (text.startsWith('response')) return 'response';
            return '';
        }
        
        function formatTrace(text) {
            // Split into blocks
            const lines = text.split('\\n');
            let blocks = [];
            let currentBlock = [];
            let braceCount = 0;
            
            for (const line of lines) {
                currentBlock.push(line);
                braceCount += (line.match(/\\{/g) || []).length;
                braceCount -= (line.match(/\\}/g) || []).length;
                
                // Check if we completed a block
                if (braceCount === 0 && currentBlock.length > 0) {
                    const blockText = currentBlock.join('\\n').trim();
                    if (blockText) {
                        blocks.push(blockText);
                    }
                    currentBlock = [];
                }
            }
            
            // Handle remaining content
            if (currentBlock.length > 0) {
                const blockText = currentBlock.join('\\n').trim();
                if (blockText) {
                    blocks.push(blockText);
                }
            }
            
            // Render blocks
            return blocks.map(block => {
                const type = getBlockType(block);
                const highlighted = highlight(block);
                return `<div class="block ${type}"><pre>${highlighted}</pre></div>`;
            }).join('');
        }
        
        function renderTraceList(filter = '') {
            const list = document.getElementById('trace-list');
            const filterLower = filter.toLowerCase();
            
            list.innerHTML = traces.map((trace, i) => {
                const traceText = trace.trace || '';
                if (filter && !traceText.toLowerCase().includes(filterLower)) {
                    return '';
                }
                
                // Detect characteristics
                const hasTools = traceText.includes('tool {');
                const hasError = traceText.toLowerCase().includes('error:');
                const isMultiTurn = (traceText.match(/user「/g) || []).length > 1;
                const lines = traceText.split(/\\r?\\n/).length;
                
                const tags = [];
                if (hasTools) tags.push('<span class="tag tools">tools</span>');
                else tags.push('<span class="tag no-tools">no-tools</span>');
                if (isMultiTurn) tags.push('<span class="tag multi-turn">multi-turn</span>');
                if (hasError) tags.push('<span class="tag error">error</span>');
                
                return `
                    <div class="trace-item ${i === currentIndex ? 'active' : ''}" onclick="selectTrace(${i})">
                        <div class="name">trace_${String(i).padStart(4, '0')}</div>
                        <div class="meta">${lines} lines</div>
                        <div class="tags">${tags.join('')}</div>
                    </div>
                `;
            }).join('');
        }
        
        function selectTrace(index) {
            currentIndex = index;
            const trace = traces[index];
            
            // Update sidebar
            document.querySelectorAll('.trace-item').forEach((el, i) => {
                el.classList.toggle('active', i === index);
            });
            
            // Update toolbar
            document.getElementById('toolbar-title').textContent = `trace_${String(index).padStart(4, '0')}`;
            const chars = trace.trace.length;
            const lines = trace.trace.split(/\\r?\\n/).length;
            document.getElementById('toolbar-info').textContent = `${lines} lines • ${chars.toLocaleString()} chars`;
            
            // Update content
            document.getElementById('trace-view').innerHTML = formatTrace(trace.trace);
        }
        
        // Initialize
        document.getElementById('stats').textContent = `${traces.length} traces`;
        renderTraceList();
        
        // Search
        document.getElementById('search').addEventListener('input', (e) => {
            renderTraceList(e.target.value);
        });
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            
            if (e.key === 'ArrowDown' || e.key === 'j') {
                e.preventDefault();
                if (currentIndex < traces.length - 1) selectTrace(currentIndex + 1);
            } else if (e.key === 'ArrowUp' || e.key === 'k') {
                e.preventDefault();
                if (currentIndex > 0) selectTrace(currentIndex - 1);
            }
        });
        
        // Select first trace
        if (traces.length > 0) selectTrace(0);
    </script>
</body>
</html>
'''


class TraceViewerHandler(SimpleHTTPRequestHandler):
    """Custom handler for the trace viewer."""
    
    traces_data = []
    
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Inject traces data into HTML
            html = HTML_TEMPLATE.replace(
                'TRACES_DATA',
                json.dumps(self.traces_data)
            )
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging


def main():
    parser = argparse.ArgumentParser(description="View TASK traces in browser")
    parser.add_argument(
        "file",
        nargs="?",
        default="traces.jsonl",
        help="Path to traces JSONL file (default: traces.jsonl)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to run server on (default: 8080)"
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    file_path = Path(args.file)
    
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return
    
    # Load traces
    print(f"Loading traces from {file_path}...")
    traces = []
    with open(file_path) as f:
        for line in f:
            try:
                traces.append(json.loads(line))
            except:
                pass
    
    print(f"Loaded {len(traces)} traces")
    
    if not traces:
        print("No traces found")
        return
    
    # Set up handler
    TraceViewerHandler.traces_data = traces
    
    # Start server
    server = HTTPServer(('localhost', args.port), TraceViewerHandler)
    url = f"http://localhost:{args.port}"
    
    print(f"\nViewer running at {url}")
    print("Press Ctrl+C to stop\n")
    
    if not args.no_open:
        webbrowser.open(url)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.shutdown()


if __name__ == "__main__":
    main()

