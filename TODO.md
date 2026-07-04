# TODO
- [x] fix: there is a bug for multiple inputs going to the same node. they should be added iff they have the same size. if not, error.
- [x] feat: in_ch or in_feature can be excluded from parameters since we ALREADY KNOW THE SHAPES.
- [x] wire up diagnostics.
- [x] node graph visualization
- [ ] optional: optimize semantic analysis
- [ ] optional: remove duplicate errors for both not reachable and not defined node
- [ ] optional: this code might be the most readable and cleanest, but it is unoptimized af.
- [ ] very optional: the generated forward def code defines too many variables. optimize it.