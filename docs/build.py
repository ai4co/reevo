import time
import os
from os.path import join
import shutil
import tomllib
import traceback

from jinja2 import Environment, FileSystemLoader

env = Environment(
    loader=FileSystemLoader("./template"),
    autoescape=False,
    auto_reload=True,
)

def build(preview=False, target_folder="./dist", config_path="./config.toml", minify = True):
    with open(config_path, 'rb') as config_file:
        config = tomllib.load(config_file)
        config['preview_mode']['enabled'] = preview
        # pprint(config)

    page = env.get_template('index.html').render(config)

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    else:
        os.system(f"rm -r '{target_folder}'")
        os.mkdir(target_folder)

    with open(join(target_folder, "index.html"), 'w') as f:
        f.write(page)

    shutil.copytree("./static", join(target_folder, "static"))

    for _, file in config.get("static_file", {}).items():
        dest = join(target_folder, file['dest'])
        directory = os.path.dirname(dest)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        if 'src' in file:
            shutil.copy(file['src'], dest)
        elif 'content' in file:
            content = file['content']
            with open(dest, 'w') as f:
                f.write(content)

    if preview:
        with open(join(target_folder, config['preview_mode']['dummy_file_path']), 'w') as f:
            f.write("ok")
    if minify:
        for path in all_filepaths(target_folder):
            filename = os.path.basename(path)
            extension = filename.rsplit(".",1)[-1].lower()
            if extension == "css":
                minify_css(path)
            elif extension in {'html', 'svg'}:
                minify_html(path)

    print("Built at", time.strftime("%Y-%m-%d %H:%M:%S"))

def start_server_daemon(ip="127.0.0.1", port=8123, directory="./dist"):
    from threading import Thread
    import http.server
    from http import HTTPStatus
    import socketserver
    
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args):
            super().__init__(*args, directory=directory)

        def log_request(self, code='-', size='-'):
            if isinstance(code, HTTPStatus):
                code = code.value
            if code != 304:  # prevent logging requests with status code 304
                return super().log_request(code, size)

    def start_server():
        with socketserver.TCPServer((ip, port), Handler) as httpd:
            print(f"Serving live preview at http://{ip}:{port}/")
            httpd.serve_forever()

    t = Thread(target=start_server, daemon=True)
    t.start()


def build_on_change(*paths, **build_kwargs):
    last_update = 0.0
    while 1:
        try:
            for filepath in all_filepaths(*paths):
                last_modified_at = os.stat(filepath).st_mtime
                if last_modified_at >= last_update:
                    build(**build_kwargs)
                    last_update = time.time()
                    break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            traceback.print_exc()
        finally:
            time.sleep(1)


def all_filepaths(*paths):
    for path in paths:
        if not os.path.exists(path):
            pass
        elif os.path.isfile(path):
            yield path
        else:
            for root, _, files in os.walk(path):
                yield root
                yield from (join(root, file) for file in files)

def minify_css(filepath):
    import re
    content = open(filepath,'r').read()
    minimized = re.sub(r' *\n *|/\*.*?\*/', '', content)
    minimized = re.sub(r'; *(?=})|(?<=:) +| +(?={)', '', minimized)
    with open(filepath, 'w') as f:
        f.write(minimized)
    print(filepath+":", f'reduced from {len(content)} Bytes to {len(minimized)} Bytes')

def minify_html(filepath):
    from htmlmin.main import minify
    content = open(filepath,'r').read()
    minimized = minify(content, remove_comments=True, remove_all_empty_space=True)
    with open(filepath, 'w') as f:
        f.write(minimized)
    print(filepath+":", f'reduced from {len(content)} Bytes to {len(minimized)} Bytes')

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--live-preview", action='store_true', help="Start live preview server")
    parser.add_argument("-i", "--ip", default="127.0.0.1", help="Preview server ip, default to 127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=8123, help="Preview server port, default to 8123")
    parser.add_argument("-o", "--output", type=Path, default="./dist", help="Build file destination, default to ./dist")
    parser.add_argument("-c", "--config", type=Path, default="./config.toml", help="Config file path, default to ./config.toml")
    parser.add_argument("-m", "--minify", default=None, action='store_true', 
                        help="Minify the output files if possible, default to True when live_preview is disabled, otherwise False")
    opts = parser.parse_args()

    opts.minify = (not opts.live_preview) if opts.minify is None else opts.minify
    build_kwargs = dict(preview=opts.live_preview, target_folder=opts.output, config_path=opts.config, minify = opts.minify)
    if opts.live_preview:
        start_server_daemon(ip=opts.ip, port=opts.port, directory=opts.output)
        build_on_change("./template", opts.config, "./static",  **build_kwargs)
    else:
        build(**build_kwargs)
