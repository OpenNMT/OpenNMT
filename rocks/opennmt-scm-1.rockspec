package = "opennmt"
version = "scm-1"

source = {
   url = "git://github.com/opennmt/opennmt",
   tag = "master"
}

description = {
   summary = "Neural Machine Translation for Torch",
   homepage = "https://github.com/opennmt/opennmt",
   license = "MIT"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "nngraph"
}

build = {
   type = "command",
   build_command = [[
      cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE)
         ]],
   install_command = "cd build && $(MAKE) install"
}