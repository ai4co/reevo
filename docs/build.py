from http import HTTPStatus
from jinja2 import Environment, FileSystemLoader
from htmlmin.main import minify
import tomllib
import os
from os.path import join
import shutil
import argparse
import time

target_folder = "./dist"

env = Environment(
    loader=FileSystemLoader("./template"),
    autoescape=False,
    auto_reload=True,
)


def build(preview=False):
    with open("./config.toml", 'rb') as config_file:
        config = tomllib.load(config_file)
        config['preview_mode']['enabled'] = preview
        # pprint(config)

    page = env.get_template('index.html').render(config)
    page = minify(page, remove_comments=True, remove_all_empty_space=True)

    if not os.path.isdir(target_folder):
        os.makedirs(target_folder, exist_ok=True)
    else:
        os.system(f"rm -r '{target_folder}'")
        os.mkdir(target_folder)

    with open(join(target_folder, "index.html"), 'w') as f:
        f.write(page)

    shutil.copytree("./static", join(target_folder, "static"))

    for key, file in config.get("static_file", {}).items():
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

    print("Built at", time.strftime("%y-%m-%d %H:%M:%S"))


def start_server_daemon(ip="127.0.0.1", port=8123):
    import http.server
    import socketserver
    from functools import partial
    from threading import Thread

    def start_server():
        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args):
                super().__init__(*args, directory=target_folder)

            def log_request(self, code='-', size='-'):
                if isinstance(code, HTTPStatus):
                    code = code.value
                if code != 304:  # filter out requests to dummy page with status code 304
                    return super().log_request(code, size)

        with socketserver.TCPServer((ip, port), Handler) as httpd:
            print(f"Serving live preview at http://{ip}:{port}/")
            httpd.serve_forever()

    t = Thread(target=start_server, daemon=True)
    t.start()


def build_on_change(*paths):
    build(preview=True)
    last_update = time.time()
    while 1:
        for filepath in all_filepaths(paths):
            last_modified_at = os.stat(filepath).st_mtime
            if last_modified_at >= last_update:
                build(preview=True)
                last_update = time.time()
                break
        else:
            time.sleep(1)


def all_filepaths(paths):
    for path in paths:
        if not os.path.exists(path):
            pass
        elif os.path.isfile(path):
            yield path
        else:
            for root, _, files in os.walk(path):
                yield root
                yield from (join(root, file) for file in files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--live-preview", action='store_true')
    parser.add_argument("-i", "--ip", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=8123)
    opts = parser.parse_args()

    if opts.live_preview:
        start_server_daemon(ip=opts.ip, port=opts.port)
        build_on_change("./template", "config.toml", "./static")
    else:
        build()
