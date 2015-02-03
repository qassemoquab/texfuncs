package = "texfuncs"
version = "scm-1"

source = {
   url = "git://github.com/qassemoquab/texfuncs",
   tag = "master"
}

description = {
   summary = "Extract-interpolate",
   detailed = [[
   	    Texture-based functions
   ]],
   homepage = "https://github.com/qassemoquab/texfuncs"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
