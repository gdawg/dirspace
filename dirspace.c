#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <errno.h>
#if defined(__linux__)
  #define __USE_BSD 1
  #include <linux/limits.h>
  extern char *program_invocation_short_name;
  static const char *getprogname(void) { return program_invocation_short_name; }
#elif defined(__APPLE__)
  #include <limits.h>
#endif
#include <sys/types.h>
#include <sys/ttydefaults.h>
#include <sys/stat.h>
#include <getopt.h>
#include <assert.h>
#include <fts.h>

#define RB_COMPACT
#include "rb.h"
#include "humanize.h"
#include "xmalloc.h"

#define GRAPH_W 40
#define BAR_CHAR u8"\u2588"
#define STREQ(S1,S2) (strcmp(S1,S2) == 0)

#define GET_OPTS \
  X(summary, no_argument, 's', int, 0, 1) \
  X(maxdepth, required_argument, 'd', int, 10, atoi(optarg)) \
  X(nofollow, no_argument, 'x', int, 0, 1) \
  X(exclusive, no_argument, 'e', int, 0, 1) \
  X(showall, required_argument, 'a', int, 0, 1)

#include "xopts.h"

/* path size calculation */
typedef struct {
  char *path;
  size_t size;
  size_t ownsize;
} dirsz_t;

typedef void(sizecb_t)(dirsz_t dirsz, void *udata);

#include <ctype.h>
#include <errno.h>

int
read_sizes(char **dirs, sizecb_t *cb, void *ud)
{
  FTS *fts;
  FTSENT *ent;
  size_t dsize, szstack[PATH_MAX], last;
  int stackd;

  fts = fts_open(dirs, FTS_PHYSICAL | (options.nofollow ? 0 : FTS_XDEV), NULL);
  if (!fts)
  {
    perror(NULL);
    exit(errno ? errno : EXIT_FAILURE);
  }

  stackd = dsize = last = 0;
  dirsz_t dirsz;
  for (ent = fts_read(fts); ent; ent = fts_read(fts))
  {
    switch (ent->fts_info)
    {
    case FTS_D:
      dirsz.ownsize = dsize;
      szstack[stackd++] = dsize;
      dsize = 0;
      break;
    case FTS_F:
      dsize += ent->fts_statp->st_size;
      break;
    case FTS_DP:
      if ((dsize != last && dsize > 0) && ent->fts_level < options.maxdepth) {
        dirsz.size = options.exclusive ? dirsz.ownsize : dsize;
        dirsz.path = ent->fts_path;

        (*cb)(dirsz, ud);
        last = dsize;
      }
      assert(stackd > 0);
      dsize += szstack[--stackd];
      break;

    default:
      break;
    }
  }

  fts_close(fts);
  return 0;
}

/* dump_sizes: dump directory size info as CSV data */
static void
dump_cb(dirsz_t dirsz, void *udata)
{
  printf("%s", dirsz.path);
  printf(",%lu", (unsigned long) dirsz.size);
  printf(",%s\n", human_size(dirsz.size));
}

static int
dump_sizes(char **dirs)
{
  return read_sizes(dirs, dump_cb, NULL);
}

/* red-black tree setup for summary info */
typedef struct node_s node_t;
struct node_s {
  char path[PATH_MAX];
  size_t size;
  size_t ownsize;

  rb_node(node_t) link;
};
typedef rb_tree(node_t) tree_t;

static int 
node_cmp(node_t *a, node_t *b)
{
  if (a->size == b->size)
    return strcmp(a->path, b->path);
  return a->size < b->size ? 1 : -1;
}
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
rb_gen(static, sz_, tree_t, node_t, link, node_cmp)
#pragma clang diagnostic pop

#define rb_destroy(a_prefix, tree) \
 do { \
  void *p; \
  for (p = a_prefix ## first(tree) ; p ; p = a_prefix ## first(tree)) \
  { \
    a_prefix ## remove(tree, p); \
    free(p);     \
  } \
 } while (0)

/* summary generation is done in two passes. cb 1 (this) builds the tree */
static void
insert_node(dirsz_t dirsz, void *udata)
{
  tree_t *ptree = udata;
  assert(ptree);

  node_t *n = xmalloc(sizeof(node_t));
  strncpy(n->path, dirsz.path, PATH_MAX);
  n->size = dirsz.size;
  n->ownsize = dirsz.ownsize;
  sz_insert(ptree, n);
}

/* cb 2 (this) dumps info */
static node_t *
summarize_node(tree_t *t, node_t *n, void *ud)
{
  static int done = 0;
  if (done)
    return NULL;

  static size_t maxsz = 0;
  if (maxsz == 0)
    maxsz = n->size;

  int blocks = GRAPH_W * n->size / maxsz;
  if (!blocks) {
    done = 1;
    return NULL;
  }

  putchar(' ');
  for (int i=0; i<blocks; i++)
    printf(BAR_CHAR);
  printf("%*s", GRAPH_W - blocks, "");
  printf(" %s", human_size(n->size));
  printf(" %s\n", n->path);

  return NULL;
}

int
show_summary(char **dirs)
{
  tree_t tree;
  node_t *n;

  memset(&tree, 0, sizeof(tree));
  sz_new(&tree);
  read_sizes(dirs, insert_node, &tree);

  n = NULL;
  do {
    n = sz_iter(&tree, n, summarize_node, NULL);
  } while (n);

  rb_destroy(sz_, &tree);
  return 0;
}

int
main(int argc, char *argv[])
{
  parse_options(argc, argv);
  argc -= optind;
  argv += optind;

  if (argc)
  {
    if (options.summary)
      return show_summary(argv);
    return dump_sizes(argv);
  }
  printf(
    "usage: %s [options] dir [dir-2 .. dir-n]\n"
    "options:\n", getprogname());
  dump_options();
  puts(
    "\n"
    // BEGIN HELP TEXT
    "A tool for visualizing directory sizes.\n"
    "\n"
    "-s  Use Summary output format designed for interactive use rather than recording precise information.\n"
    "-x  Prevent descending into directories that have a device number different than that of the file from which the descent began.\n"
    // END HELP TEXT
  );
  return opterr ? opterr : EXIT_FAILURE;
}
