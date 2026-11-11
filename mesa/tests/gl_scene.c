/* SPDX-License-Identifier: Apache-2.0
 *
 * Deterministic fixed-function OpenGL test scene for the softpipe fast path (mesa/README.md).
 *
 *   gl_scene [--glsl|--glsl-alu] [--frames N] [--size S] [--dump PREFIX] [--blend|--blend-sep] [--depthmask] [--nodepth] [--lighting] [--cull] [--swap] [--ortho] [--depth] [--continue]
 *
 * Draws N frames (rotation angle = frame index), reads every frame back with glReadPixels and, with --dump,
 * writes PREFIX<frame>.ppm. --blend draws one extra alpha-blended triangle; --blend-sep the same with separate alpha
 * factors (a draw the fast path refuses: tests the mid-frame fallback); --depthmask adds additive layers drawn with depth
 * writes off, then an opaque triangle behind them; --nodepth an overlay drawn with the depth test off; --lighting uses fixed-function lighting; --cull enables back-face culling; --swap also
 * calls glXSwapBuffers every frame; --glsl draws with a GLSL 1.20 program (lighting in the vertex shader, the fragment
 * shader passes the colour through: the fast path takes it), --glsl-alu has a fragment shader with arithmetic (the fast
 * path refuses it: softpipe draws); --ortho uses an orthographic projection (w = 1); --depth also dumps the depth
 * buffer (PREFIX<frame>_depth.pgm, 16 bit); --continue draws one more triangle after the read-back, without a clear
 * (it needs the depth buffer of the frame so far) and dumps PREFIX<frame>b.ppm.
 */
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void
tri(float x0, float y0, float z0, float r0, float g0, float b0,
    float x1, float y1, float z1, float r1, float g1, float b1,
    float x2, float y2, float z2, float r2, float g2, float b2)
{
   glBegin(GL_TRIANGLES);
   glColor3f(r0, g0, b0); glVertex3f(x0, y0, z0);
   glColor3f(r1, g1, b1); glVertex3f(x1, y1, z1);
   glColor3f(r2, g2, b2); glVertex3f(x2, y2, z2);
   glEnd();
}

static GLuint
make_program(int alu)
{
   static const char *vs =
      "#version 120\n"
      "varying vec4 c;\n"
      "void main() {\n"
      "  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n"
      "  vec3 n = normalize(gl_NormalMatrix * gl_Normal);\n"
      "  float d = max(dot(n, normalize(vec3(1.0, 2.0, 3.0))), 0.0);\n"
      "  c = vec4(gl_Color.rgb * (0.4 + 0.6 * d), gl_Color.a);\n"
      "}\n";
   static const char *fs_pass = "#version 120\nvarying vec4 c;\nvoid main() { gl_FragColor = c; }\n";
   static const char *fs_alu = "#version 120\nvarying vec4 c;\nvoid main() { gl_FragColor = vec4(c.rgb * 0.5 + vec3(0.25), 1.0); }\n";
   GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER), p = glCreateProgram();
   const char *fsrc = alu ? fs_alu : fs_pass;
   glShaderSource(v, 1, &vs, NULL);
   glShaderSource(f, 1, &fsrc, NULL);
   glCompileShader(v);
   glCompileShader(f);
   glAttachShader(p, v);
   glAttachShader(p, f);
   glLinkProgram(p);
   GLint ok = 0;
   glGetProgramiv(p, GL_LINK_STATUS, &ok);
   if (!ok) { fprintf(stderr, "GLSL link failed\n"); exit(1); }
   return p;
}

static void
draw_scene(int frame, int blend, int lighting, int ortho, GLuint program, int depthmask, int nodepth)
{
   glClearColor(0.1f, 0.12f, 0.2f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   glEnable(GL_DEPTH_TEST);
   glDepthFunc(GL_LESS);

   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   if (ortho)
      glOrtho(-3.0, 3.0, -3.0, 3.0, 1.0, 20.0);   /* w = 1: perspective-correct = screen-linear */
   else
      glFrustum(-1.0, 1.0, -1.0, 1.0, 2.0, 20.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   glTranslatef(0.0f, 0.0f, -6.0f);
   glRotatef(12.0f * frame, 0.3f, 1.0f, 0.2f);

   if (program) glUseProgram(program);
   if (lighting) {
      const GLfloat pos[4] = {1.0f, 2.0f, 3.0f, 0.0f};
      glEnable(GL_LIGHTING);
      glEnable(GL_LIGHT0);
      glLightfv(GL_LIGHT0, GL_POSITION, pos);
      glEnable(GL_COLOR_MATERIAL);
      glEnable(GL_NORMALIZE);
   }

   /* Two large triangles that pierce each other (depth interleaving). */
   glShadeModel(GL_SMOOTH);
   if (lighting || program) glNormal3f(0.0f, 0.0f, 1.0f);
   tri(-2.0f, -1.5f, -1.0f, 1.0f, 0.1f, 0.1f,   2.0f, -1.0f, 1.0f, 0.1f, 1.0f, 0.1f,   0.0f, 2.0f, 0.0f, 0.1f, 0.1f, 1.0f);
   tri(-1.5f, 1.5f, 1.0f, 1.0f, 1.0f, 0.1f,   1.5f, 1.2f, -1.0f, 0.1f, 1.0f, 1.0f,   0.0f, -2.0f, 0.5f, 1.0f, 0.1f, 1.0f);

   /* Flat-shaded quad (two triangles, colour of the provoking vertex). */
   glShadeModel(GL_FLAT);
   glBegin(GL_QUADS);
   glColor3f(0.9f, 0.5f, 0.1f); glVertex3f(-0.8f, -0.8f, 0.6f);
   glColor3f(0.2f, 0.9f, 0.5f); glVertex3f(0.8f, -0.8f, 0.6f);
   glColor3f(0.5f, 0.2f, 0.9f); glVertex3f(0.8f, 0.8f, -0.4f);
   glColor3f(0.9f, 0.9f, 0.9f); glVertex3f(-0.8f, 0.8f, -0.4f);
   glEnd();

   /* Smooth fan and strip. */
   glShadeModel(GL_SMOOTH);
   glBegin(GL_TRIANGLE_FAN);
   glColor3f(1.0f, 1.0f, 1.0f); glVertex3f(1.5f, -1.5f, 0.2f);
   for (int i = 0; i <= 16; i++) {
      const float a = (float)i * (6.2831853f / 16.0f);
      glColor3f(0.5f + 0.5f * cosf(a), 0.5f + 0.5f * sinf(a), 0.3f);
      glVertex3f(1.5f + 0.7f * cosf(a), -1.5f + 0.7f * sinf(a), 0.2f);
   }
   glEnd();
   glBegin(GL_TRIANGLE_STRIP);
   for (int i = 0; i < 10; i++) {
      glColor3f(0.2f + 0.08f * i, 0.8f - 0.07f * i, 0.5f);
      glVertex3f(-2.6f + 0.25f * i, (i & 1) ? -2.2f : -1.7f, -0.3f);
   }
   glEnd();

   if (blend) {
      glUseProgram(0);
      glDisable(GL_LIGHTING);
      glEnable(GL_BLEND);
      if (blend == 2)
         glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ZERO);   /* separate alpha: refused */
      else
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glBegin(GL_TRIANGLES);
      glColor4f(1.0f, 1.0f, 1.0f, 0.4f); glVertex3f(-1.0f, -1.0f, 1.5f);
      glColor4f(1.0f, 0.0f, 0.0f, 0.4f); glVertex3f(1.2f, -0.8f, 1.5f);
      glColor4f(0.0f, 0.0f, 1.0f, 0.4f); glVertex3f(0.0f, 1.3f, 1.5f);
      glEnd();
      glDisable(GL_BLEND);
   }

   if (depthmask) {
      /* Translucent layers that test depth but do not write it, then an opaque triangle behind them. */
      glUseProgram(0);
      glDisable(GL_LIGHTING);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE);
      glDepthMask(GL_FALSE);
      for (int k = 0; k < 3; k++) {
         glBegin(GL_TRIANGLES);
         glColor4f(1.0f, 0.5f, 0.1f, 0.35f); glVertex3f(-1.4f + 0.3f * k, -1.2f, 0.9f - 0.2f * k);
         glColor4f(0.1f, 0.6f, 1.0f, 0.35f); glVertex3f(1.4f + 0.3f * k, -0.9f, 0.9f - 0.2f * k);
         glColor4f(1.0f, 1.0f, 1.0f, 0.35f); glVertex3f(0.1f * k, 1.6f, 0.2f);
         glEnd();
      }
      glDepthMask(GL_TRUE);
      glDisable(GL_BLEND);
      glBegin(GL_TRIANGLES);
      glColor3f(0.9f, 0.9f, 0.2f); glVertex3f(-2.4f, -2.0f, -0.8f);
      glColor3f(0.2f, 0.9f, 0.9f); glVertex3f(2.4f, -1.9f, -0.8f);
      glColor3f(0.9f, 0.2f, 0.9f); glVertex3f(0.0f, 2.4f, -0.8f);
      glEnd();
   }

   if (nodepth) {
      /* An overlay drawn with the depth test off (and so without touching the depth buffer). */
      glUseProgram(0);
      glDisable(GL_LIGHTING);
      glDisable(GL_DEPTH_TEST);
      glBegin(GL_TRIANGLES);
      glColor3f(0.0f, 0.0f, 0.0f); glVertex3f(-0.6f, -0.6f, 2.0f);
      glColor3f(1.0f, 1.0f, 1.0f); glVertex3f(0.6f, -0.5f, 2.0f);
      glColor3f(0.5f, 0.0f, 0.5f); glVertex3f(0.0f, 0.7f, 2.0f);
      glEnd();
      glEnable(GL_DEPTH_TEST);
   }
}

static void
dump_rgba(const char *prefix, int frame, const char *suffix, const unsigned char *px, int size)
{
   char name[512];
   snprintf(name, sizeof name, "%s%d%s.ppm", prefix, frame, suffix);
   FILE *fp = fopen(name, "wb");
   fprintf(fp, "P6\n%d %d\n255\n", size, size);
   for (int y = size - 1; y >= 0; y--)   /* top row first */
      for (int x = 0; x < size; x++)
         fwrite(&px[((size_t)y * size + x) * 4], 1, 3, fp);
   fclose(fp);
}

static void
dump_depth(const char *prefix, int frame, int size)
{
   float *z = malloc((size_t)size * size * sizeof(float));
   glReadPixels(0, 0, size, size, GL_DEPTH_COMPONENT, GL_FLOAT, z);
   char name[512];
   snprintf(name, sizeof name, "%s%d_depth.pgm", prefix, frame);
   FILE *fp = fopen(name, "wb");
   fprintf(fp, "P5\n%d %d\n65535\n", size, size);
   for (int y = size - 1; y >= 0; y--)
      for (int x = 0; x < size; x++) {
         const unsigned v = (unsigned)(z[(size_t)y * size + x] * 65535.0f + 0.5f);
         fputc((v >> 8) & 255, fp);
         fputc(v & 255, fp);
      }
   fclose(fp);
   free(z);
}

int
main(int argc, char **argv)
{
   int frames = 1, size = 256, blend = 0, lighting = 0, cull = 0, swap = 0, ortho = 0, depth_out = 0, cont = 0, glsl = 0, glsl_alu = 0, depthmask = 0, nodepth = 0;
   const char *dump = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--size") && i + 1 < argc) size = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
      else if (!strcmp(argv[i], "--blend")) blend = 1;
      else if (!strcmp(argv[i], "--blend-sep")) blend = 2;
      else if (!strcmp(argv[i], "--depthmask")) depthmask = 1;
      else if (!strcmp(argv[i], "--nodepth")) nodepth = 1;
      else if (!strcmp(argv[i], "--lighting")) lighting = 1;
      else if (!strcmp(argv[i], "--cull")) cull = 1;
      else if (!strcmp(argv[i], "--swap")) swap = 1;
      else if (!strcmp(argv[i], "--ortho")) ortho = 1;
      else if (!strcmp(argv[i], "--depth")) depth_out = 1;
      else if (!strcmp(argv[i], "--glsl")) glsl = 1;
      else if (!strcmp(argv[i], "--glsl-alu")) glsl_alu = 1;
      else if (!strcmp(argv[i], "--continue")) cont = 1;
   }
   Display *dpy = XOpenDisplay(NULL);
   if (!dpy) { fprintf(stderr, "no X display\n"); return 1; }
   int attr[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 24, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
   XVisualInfo *vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
   if (!vi) { fprintf(stderr, "no visual\n"); return 1; }
   XSetWindowAttributes swa = {0};
   swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
   swa.event_mask = StructureNotifyMask;
   Window win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, size, size, 0, vi->depth, InputOutput, vi->visual,
                              CWColormap | CWEventMask, &swa);
   XMapWindow(dpy, win);
   GLXContext ctx = glXCreateContext(dpy, vi, NULL, True);
   glXMakeCurrent(dpy, win, ctx);
   fprintf(stderr, "GL_RENDERER = %s\n", (const char *)glGetString(GL_RENDERER));
   glViewport(0, 0, size, size);
   if (cull) { glEnable(GL_CULL_FACE); glCullFace(GL_BACK); }

   const GLuint program = glsl || glsl_alu ? make_program(glsl_alu) : 0;
   unsigned char *px = malloc((size_t)size * size * 4);
   for (int f = 0; f < frames; f++) {
      draw_scene(f, blend, lighting, ortho, program, depthmask, nodepth);
      glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE, px);
      if (dump) {
         dump_rgba(dump, f, "", px, size);
         if (depth_out) dump_depth(dump, f, size);
      }
      if (cont) {
         /* No clear: the triangle is depth-tested against what the frame drew. */
         glUseProgram(0);
         glDisable(GL_LIGHTING);
         glEnable(GL_DEPTH_TEST);
         glBegin(GL_TRIANGLES);
         glColor3f(1.0f, 1.0f, 1.0f); glVertex3f(-2.5f, -0.3f, 0.35f);
         glColor3f(0.0f, 0.0f, 0.0f); glVertex3f(2.5f, -0.3f, -0.35f);
         glColor3f(1.0f, 0.0f, 1.0f); glVertex3f(0.0f, 2.2f, 0.0f);
         glEnd();
         glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE, px);
         if (dump) {
            dump_rgba(dump, f, "b", px, size);
            if (depth_out) dump_depth(dump, f, size);
         }
      }
      if (swap) glXSwapBuffers(dpy, win);
   }
   free(px);
   glXMakeCurrent(dpy, None, NULL);
   glXDestroyContext(dpy, ctx);
   XCloseDisplay(dpy);
   return 0;
}
