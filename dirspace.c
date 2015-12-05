#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <libgen.h>
#include <ctype.h>
#include <getopt.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <limits.h>
#include <fts.h>

#define verbose(...) if (isverbose) printf(__VAR_ARGS__)

static int isverbose = 0;
static char sepchar = ',';

static int
fatal_err(char *msg)
{
  perror(msg);
  return EXIT_FAILURE;
}

typedef struct unit_s {
  const char *name;
  size_t size;
} unit_t;

static const unit_t units[] = {
  {"b", 1},
  {"k", 1024},
  {"meg", 1024 * 1024},
  {"gig", 1024 * 1024 * 1024},
};
#define ARRAY_SIZE(a) (sizeof(a) / sizeof(*a))

static unit_t
human_unit(size_t size)
{
  int i;
  for (i=1; i<ARRAY_SIZE(units); i++)
    if (units[i].size > size)
      break;
  i--;
  return units[i];
}

static void
showsize(FTSENT *ent, size_t size)
{
  unit_t u;

  printf("%s", ent->fts_path);
  printf("%c%lu", sepchar, (unsigned long)size);

  u = human_unit(size);
  printf("%c%lu%s", sepchar, (unsigned long)(size / u.size), u.name);

  putchar('\n');
}

static int
process(char **dirs, int xdev, int maxdepth)
{
  FTS *fts;
  FTSENT *ent;
  size_t dsize, szstack[PATH_MAX], last;
  int stackd;

  fts = fts_open(dirs, FTS_PHYSICAL | (xdev ? 0 : FTS_XDEV ), NULL);
  if (!fts)
    return fatal_err(NULL);

  stackd = dsize = last = 0;
  for (ent = fts_read(fts); ent ; ent = fts_read(fts))
  {
    switch (ent->fts_info)
    {
      case FTS_D: 
        szstack[stackd++] = dsize;
        dsize = 0;
        break;
      case FTS_F: 
        dsize += ent->fts_statp->st_size;
        break;
      case FTS_DP:
        if (isverbose || ((dsize != last && dsize > 0) && ent->fts_level < maxdepth))
        {
          showsize(ent, dsize);
          last = dsize;
        }
        assert(stackd > 0);
        dsize += szstack[--stackd];
        break;

      default: break;
    }

  }

  fts_close(fts); 
  return EXIT_SUCCESS;
}

int
main(int argc, char *argv[])
{
  int c, xdev, maxdepth;
  
  char *name = basename(argv[0]);

  static struct option longopts[] = {
    {"verbose", no_argument, NULL, 'v'},
    {"xdevice", no_argument, NULL, 'x'},
    {"separator", required_argument, NULL, 's'},
    {"maxdepth", required_argument, NULL, 'd'},
    {NULL, 0, NULL, 0}
  };

  xdev = isverbose = opterr = 0;
  maxdepth = __INT_MAX__;
  while ((c = getopt_long(argc, argv, ":vxs:d:", longopts, NULL)) != -1){
  switch(c){
    case 'v': isverbose = 1; break;
    case 'x': xdev = 1; break;
    case 'd': maxdepth = atoi(optarg); break;
    case 's': sepchar = optarg[0]; break;
    default: opterr = 1; break;
  }}
  argc -= optind;
  argv += optind;

  if (argc <= 0)
    opterr = 1;

  if (opterr)
  {
    printf(
      "usage: %s [options] dir [dir-2 .. dir-n]\n"
      "options:\n", name);
    puts(
      "    -v                --verbose\n");
    puts("Recursively calculates and summarises size of directories.");
    return EXIT_FAILURE;
  }

  return process(argv, xdev, maxdepth);
}
