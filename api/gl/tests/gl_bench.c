/* SPDX-License-Identifier: Apache-2.0
 *
 * A small OpenGL benchmark that exercises the parts of a software rasterizer that differ most: triangle count, overdraw and
 * fragment ALU work. One frame is a clear plus the draws below, then glFinish and a one-pixel read-back (so the frame is rendered
 * but not copied to a window).
 *
 *   gl_bench [--size WxH] [--frames N] [--tris N] [--layers L] [--alu K]
 *
 *   --tris N     N small triangles tiling the window (fixed-function, per-vertex colours, depth test on)
 *   --layers L   L full-window quads, back to front, no depth test (overdraw); with --alu K each has a GLSL fragment shader of K
 *                dependent multiply-adds (an unrolled chain, so the shader is straight-line code)
 *   --alu K      steps of that chain, two instructions each (default 0: a colour pass-through)
 *
 * Prints "ms/frame" and FPS over the timed frames (after 5 warm-up frames).
 */
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static GLuint make_program(int alu)
{
   const char *vs = "#version 120\nvarying vec2 uv;\nvoid main(){ uv = gl_Vertex.xy * 0.5 + 0.5; gl_Position = gl_Vertex; }\n";
   char *fs = malloc(64 * 1024);
   int n = snprintf(fs, 4096, "#version 120\nvarying vec2 uv;\nvoid main(){\n float x = uv.x;\n");
   for (int i = 0; i < alu; i++)
      n += snprintf(fs + n, 128, " x = fract(x * 1.37 + uv.y);\n");   /* two instructions a step; the fract keeps the GLSL optimizer from folding the chain */
   snprintf(fs + n, 256, " gl_FragColor = vec4(x, x * 0.5, uv.y, 1.0);\n}\n");
   GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER), p = glCreateProgram();
   glShaderSource(v, 1, &vs, NULL);
   glCompileShader(v);
   const char *fsp = fs;
   glShaderSource(f, 1, &fsp, NULL);
   glCompileShader(f);
   GLint ok;
   glGetShaderiv(f, GL_COMPILE_STATUS, &ok);
   if (!ok) {
      char log[2048];
      glGetShaderInfoLog(f, sizeof log, NULL, log);
      fprintf(stderr, "fragment shader: %s\n", log);
      exit(1);
   }
   glAttachShader(p, v);
   glAttachShader(p, f);
   glLinkProgram(p);
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) {
      fprintf(stderr, "link failed\n");
      exit(1);
   }
   free(fs);
   return p;
}

int main(int argc, char **argv)
{
   int w = 1920, h = 1080, frames = 100, tris = 0, layers = 0, alu = 0;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--size") && i + 1 < argc) sscanf(argv[++i], "%dx%d", &w, &h);
      else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--tris") && i + 1 < argc) tris = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--layers") && i + 1 < argc) layers = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--alu") && i + 1 < argc) alu = atoi(argv[++i]);
   }
   if (!tris && !layers) layers = 1;
   Display *dpy = XOpenDisplay(NULL);
   if (!dpy) { fprintf(stderr, "no X display\n"); return 1; }
   int attr[] = {GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None};
   XVisualInfo *vi = glXChooseVisual(dpy, 0, attr);
   if (!vi) { fprintf(stderr, "no visual\n"); return 1; }
   Window root = RootWindow(dpy, vi->screen);
   XSetWindowAttributes swa = {.colormap = XCreateColormap(dpy, root, vi->visual, AllocNone), .border_pixel = 0};
   Window win = XCreateWindow(dpy, root, 0, 0, w, h, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWBorderPixel, &swa);
   XMapWindow(dpy, win);
   GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
   glXMakeCurrent(dpy, win, ctx);
   glViewport(0, 0, w, h);
   fprintf(stderr, "GL_RENDERER = %s\n", (const char *)glGetString(GL_RENDERER));

   GLuint prog = layers ? make_program(alu) : 0;
   /* Triangle soup: a grid of cells, two triangles each, per-vertex colours, spread in depth so the depth test works. */
   float *verts = NULL, *cols = NULL;
   int ntri = tris;
   if (tris) {
      int cells = (tris + 1) / 2, n = 1;
      while (n * n < cells) n++;
      ntri = 2 * n * n;
      verts = malloc(sizeof(float) * 9 * ntri);
      cols = malloc(sizeof(float) * 9 * ntri);
      int t = 0;
      for (int cy = 0; cy < n; cy++)
         for (int cx = 0; cx < n; cx++) {
            float x0 = -1 + 2.0f * cx / n, x1 = -1 + 2.0f * (cx + 1) / n, y0 = -1 + 2.0f * cy / n, y1 = -1 + 2.0f * (cy + 1) / n;
            float z = 0.2f + 0.6f * ((cx * 7 + cy * 13) % 16) / 16.0f;
            float q[2][3][2] = {{{x0, y0}, {x1, y0}, {x1, y1}}, {{x0, y0}, {x1, y1}, {x0, y1}}};
            for (int k = 0; k < 2; k++, t++)
               for (int v = 0; v < 3; v++) {
                  verts[9 * t + 3 * v + 0] = q[k][v][0];
                  verts[9 * t + 3 * v + 1] = q[k][v][1];
                  verts[9 * t + 3 * v + 2] = z;
                  cols[9 * t + 3 * v + 0] = (q[k][v][0] + 1) * 0.5f;
                  cols[9 * t + 3 * v + 1] = (q[k][v][1] + 1) * 0.5f;
                  cols[9 * t + 3 * v + 2] = z;
               }
         }
   }
   double t0 = 0;
   unsigned char px[4];
   for (int f = 0; f < frames + 5; f++) {
      if (f == 5) t0 = now_ms();
      glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      if (tris) {
         glEnable(GL_DEPTH_TEST);
         glDepthFunc(GL_LESS);
         glUseProgram(0);
         glEnableClientState(GL_VERTEX_ARRAY);
         glEnableClientState(GL_COLOR_ARRAY);
         glVertexPointer(3, GL_FLOAT, 0, verts);
         glColorPointer(3, GL_FLOAT, 0, cols);
         glDrawArrays(GL_TRIANGLES, 0, 3 * ntri);
      } else {
         glDisable(GL_DEPTH_TEST);
         glUseProgram(prog);
         static const float quad[] = {-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, 1, 0, -1, 1, 0};
         glEnableClientState(GL_VERTEX_ARRAY);
         glDisableClientState(GL_COLOR_ARRAY);
         glVertexPointer(3, GL_FLOAT, 0, quad);
         for (int l = 0; l < layers; l++) glDrawArrays(GL_TRIANGLES, 0, 6);
      }
      glFinish();
      glReadPixels(w / 2, h / 2, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
   }
   double ms = (now_ms() - t0) / frames;
   printf("%dx%d tris=%d layers=%d alu=%d: %.2f ms/frame, %.1f FPS\n", w, h, ntri, layers, alu, ms, 1000.0 / ms);
   glXMakeCurrent(dpy, None, NULL);
   glXDestroyContext(dpy, ctx);
   XCloseDisplay(dpy);
   return 0;
}
