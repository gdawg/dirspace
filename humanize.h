#ifndef HUMANIZE_H
#define HUMANIZE_H 1

static const int kHumanBufLen = 16;

static inline char*
human_size(size_t size)
{
  static char s[kHumanBufLen];
  struct {
    const char *n; size_t sz;
  } units[] = {
    {"b", 1},
    {"k", 1024},
    {"meg", 1024 * 1024},
    {"gig", 1024 * 1024 * 1024},
  };
  int i;

  for (i=1; i<(sizeof(units) / sizeof(*units)); i++)
    if (units[i].sz > size)
      break;
  i--;

  snprintf(s, kHumanBufLen, "%lu%s", 
          (unsigned long) (size / units[i].sz), 
          units[i].n);
  return s;
}

#endif
