/* SPDX-License-Identifier: Apache-2.0
 *
 * Random fixed-function OpenGL frames for the softpipe fast path (api/gl/README.md; driven by api/gl/tests/fuzz_gl.py).
 *
 *   gl_fuzz --seed N [--size S] [--frames F] [--dump PREFIX] [--no-refused]
 *
 * Every frame starts with a full clear (random colour and depth) and then issues 3 to 14 draws, each with its own random state:
 * depth test on or off, depth function (LESS, LEQUAL, ALWAYS and, now and then, one the fast path refuses), depth writes,
 * blending (the supported factor pairs, now and then another), flat or smooth shading, face culling and winding, a primitive type
 * (triangles, strips, fans, quads, polygons) with random vertices (partly outside the window, so clipping happens), random colours
 * and alphas, random depths (surfaces cross each other), an orthographic or a perspective projection, and, rarely, state that
 * forces softpipe to take over in the middle of the frame (colour mask, scissor, line polygon mode, alpha test, stencil, ...).
 * With --no-refused all of that state is left out, so every draw is one the fast path accepts; without it, a third of the frames
 * contain one draw with such state (refused_pick), after which softpipe draws the rest of the frame.
 * The frame is read back with glReadPixels (colour, and the depth buffer to PREFIX<frame>_depth.pgm); with --dump the colour goes
 * to PREFIX<frame>.ppm. The same seed gives the same frames.
 */
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <X11/Xlib.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t g_rng;
static uint32_t
rnd(void)
{
   g_rng ^= g_rng << 13;
   g_rng ^= g_rng >> 7;
   g_rng ^= g_rng << 17;
   return (uint32_t)(g_rng >> 16);
}
static float rndf(void) { return (rnd() & 0xFFFFFF) / (float)0x1000000; }
static float range(float a, float b) { return a + (b - a) * rndf(); }
static int chance(float p) { return rndf() < p; }

static void
dump_rgba(const char *prefix, int frame, const unsigned char *px, int size)
{
   char name[512];
   snprintf(name, sizeof name, "%s%d.ppm", prefix, frame);
   FILE *fp = fopen(name, "wb");
   fprintf(fp, "P6\n%d %d\n255\n", size, size);
   for (int y = size - 1; y >= 0; y--)
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

static const GLenum fast_depth_funcs[] = {GL_LESS, GL_LEQUAL, GL_ALWAYS};
static const GLenum other_depth_funcs[] = {GL_GREATER, GL_GEQUAL, GL_EQUAL, GL_NOTEQUAL, GL_NEVER};
static const GLenum fast_blend[][2] = {
   {GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA}, {GL_ONE, GL_ONE}, {GL_SRC_ALPHA, GL_ONE}, {GL_ONE, GL_ONE_MINUS_SRC_ALPHA}, {GL_DST_COLOR, GL_ZERO},
};
static const GLenum other_blend[][2] = {
   {GL_ZERO, GL_ONE}, {GL_ONE_MINUS_DST_COLOR, GL_ZERO}, {GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA}, {GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA},
};

static int g_trace;
static int g_wild = 1;   /* --tame: moderate triangles and an orthographic projection (softpipe's own precision is poor for huge perspective triangles) */

static void
draw_one(int refused_ok, int *forced, int issue)
{
   /* State. */
   glDisable(GL_SCISSOR_TEST);
   glDisable(GL_ALPHA_TEST);
   glDisable(GL_STENCIL_TEST);
   glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
   glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
   *forced = 0;

   /* At most one refused feature per draw, and only on the draws the caller picked. */
   const int refused_pick = refused_ok ? rnd() % 8 : -1;
   char desc[256];
   int len = 0;
   if (chance(0.85f)) {
      glEnable(GL_DEPTH_TEST);
      GLenum fn;
      if (refused_pick == 0) { fn = other_depth_funcs[rnd() % 5]; *forced = 1; }
      else fn = fast_depth_funcs[rnd() % 3];
      glDepthFunc(fn);
      len += snprintf(desc + len, sizeof desc - len, "depth 0x%x ", fn);
   } else {
      glDisable(GL_DEPTH_TEST);
      len += snprintf(desc + len, sizeof desc - len, "no depth test ");
   }
   const int dmask = chance(0.75f);
   glDepthMask(dmask ? GL_TRUE : GL_FALSE);
   len += snprintf(desc + len, sizeof desc - len, "dmask %d ", dmask);
   if (chance(0.4f) || refused_pick == 1) {
      glEnable(GL_BLEND);
      const GLenum *b = refused_pick == 1 ? other_blend[rnd() % 4] : fast_blend[rnd() % 5];
      if (refused_pick == 1) *forced = 1;
      glBlendFunc(b[0], b[1]);
      len += snprintf(desc + len, sizeof desc - len, "blend 0x%x,0x%x ", b[0], b[1]);
   } else {
      glDisable(GL_BLEND);
   }
   const int flat = chance(0.2f);
   glShadeModel(flat ? GL_FLAT : GL_SMOOTH);
   len += snprintf(desc + len, sizeof desc - len, flat ? "flat " : "smooth ");
   if (chance(0.25f)) {
      glEnable(GL_CULL_FACE);
      const GLenum face = chance(0.5f) ? GL_BACK : GL_FRONT;
      glCullFace(face);
      len += snprintf(desc + len, sizeof desc - len, "cull 0x%x ", face);
   } else {
      glDisable(GL_CULL_FACE);
   }
   const int cw = !chance(0.8f);
   glFrontFace(cw ? GL_CW : GL_CCW);
   len += snprintf(desc + len, sizeof desc - len, cw ? "CW " : "CCW ");
   if (refused_pick == 2) { glColorMask(GL_TRUE, chance(0.5f), GL_TRUE, GL_TRUE); *forced = 1; }
   if (refused_pick == 3) { glEnable(GL_SCISSOR_TEST); glScissor(8, 8, 40, 48); *forced = 1; }
   if (refused_pick == 4) { glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); *forced = 1; }
   if (refused_pick == 5) { glEnable(GL_ALPHA_TEST); glAlphaFunc(GL_GREATER, 0.4f); *forced = 1; }
   if (refused_pick == 6) { glEnable(GL_STENCIL_TEST); glStencilFunc(GL_ALWAYS, 1, 0xff); *forced = 1; }

   /* Geometry. */
   const float z0 = range(-0.9f, 0.9f);
   const float tilt_x = range(-0.5f, 0.5f), tilt_y = range(-0.5f, 0.5f);   /* the surface is slanted in depth */
   const int kind = rnd() % 8;
   int n_vertices;
   GLenum mode;
   switch (kind) {
   case 0: case 1: case 2: mode = GL_TRIANGLES; n_vertices = 3 * (1 + rnd() % 4); break;
   case 3: mode = GL_TRIANGLE_STRIP; n_vertices = 4 + rnd() % 4; break;
   case 4: mode = GL_TRIANGLE_FAN; n_vertices = 4 + rnd() % 4; break;
   case 5: mode = GL_QUADS; n_vertices = 4 * (1 + rnd() % 2); break;
   case 6: mode = GL_POLYGON; n_vertices = 4 + rnd() % 3; break;
   default: mode = GL_TRIANGLES; n_vertices = 3; break;
   }
   const float cx = range(-1.0f, 1.0f), cy = range(-1.0f, 1.0f);
   const float r = g_wild ? (chance(0.2f) ? range(0.8f, 1.4f) : range(0.1f, 0.8f)) : range(0.1f, 0.6f);
   if (g_trace) fprintf(stderr, "draw: %s mode 0x%x, %d vertices\n", desc, mode, n_vertices);
   if (issue) glBegin(mode);
   for (int i = 0; i < n_vertices; i++) {
      float x, y;
      if (mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP) {
         x = cx + r * range(-1, 1);
         y = cy + r * range(-1, 1);
      } else {   /* convex position around the centre (fans, quads, polygons) */
         const float a = 6.2831853f * (i + range(-0.2f, 0.2f)) / n_vertices;
         x = cx + r * cosf(a);
         y = cy + r * sinf(a);
      }
      const float z = chance(0.3f) ? range(-1.1f, 1.1f) : z0 + tilt_x * x + tilt_y * y;
      const float cr = rndf(), cg = rndf(), cb = rndf(), ca = chance(0.5f) ? range(0.1f, 1.0f) : 1.0f;
      if (g_trace) fprintf(stderr, "  v (%.3f, %.3f, %.3f) colour (%.2f %.2f %.2f %.2f)\n", x, y, z, cr, cg, cb, ca);
      if (issue) { glColor4f(cr, cg, cb, ca); glVertex3f(x, y, z); }
   }
   if (issue) glEnd();
}

int
main(int argc, char **argv)
{
   int size = 96, frames = 2, refused_ok = 1, stop_after = 1 << 30;
   g_wild = 1;
   uint64_t seed = 1;
   const char *dump = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
      else if (!strcmp(argv[i], "--size") && i + 1 < argc) size = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--frames") && i + 1 < argc) frames = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
      else if (!strcmp(argv[i], "--no-refused")) refused_ok = 0;
      else if (!strcmp(argv[i], "--trace")) g_trace = 1;
      else if (!strcmp(argv[i], "--tame")) g_wild = 0;
      else if (!strcmp(argv[i], "--stop-after") && i + 1 < argc) stop_after = atoi(argv[++i]);   /* draws per frame actually issued (the random numbers still advance) */
   }
   g_rng = seed * 0x9E3779B97F4A7C15ull + 12345;
   for (int i = 0; i < 8; i++) rnd();
   Display *dpy = XOpenDisplay(NULL);
   if (!dpy) { fprintf(stderr, "no X display\n"); return 1; }
   int attr[] = {GLX_RGBA, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 24, GLX_STENCIL_SIZE, 8, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None};
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
   glViewport(0, 0, size, size);
   unsigned char *px = malloc((size_t)size * size * 4);
   for (int f = 0; f < frames; f++) {
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();
      if (g_wild && chance(0.4f)) {   /* perspective: the geometry sits at z in [-1, 1] in front of an eye at z = 3 */
         glMatrixMode(GL_PROJECTION);
         glFrustum(-0.4, 0.4, -0.4, 0.4, 1.0, 6.0);
         glMatrixMode(GL_MODELVIEW);
         glTranslatef(0.0f, 0.0f, -3.0f);
      } else {
         glMatrixMode(GL_PROJECTION);
         glOrtho(-1, 1, -1, 1, -1, 1);
         glMatrixMode(GL_MODELVIEW);
      }
      glDisable(GL_LIGHTING);
      glDisable(GL_SCISSOR_TEST);
      glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
      glDepthMask(GL_TRUE);
      glClearColor(rndf(), rndf(), rndf(), 1.0f);
      glClearDepth(chance(0.8f) ? 1.0 : range(0.5f, 1.0f));
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
      const int draws = 3 + rnd() % 12;
      /* With refused state allowed, a third of the frames contain one draw that the fast path cannot take. */
      const int refused_draw = refused_ok && chance(0.35f) ? (int)(rnd() % draws) : -1;
      for (int d = 0; d < draws; d++) {
         int forced;
         if (g_trace) fprintf(stderr, "frame %d draw %d\n", f, d);
         draw_one(d == refused_draw, &forced, d < stop_after);
      }
      glReadPixels(0, 0, size, size, GL_RGBA, GL_UNSIGNED_BYTE, px);
      if (dump) {
         dump_rgba(dump, f, px, size);
         dump_depth(dump, f, size);
      }
   }
   free(px);
   glXMakeCurrent(dpy, None, NULL);
   glXDestroyContext(dpy, ctx);   /* the driver reports its statistics when the context goes */
   XCloseDisplay(dpy);
   return 0;
}
