#ifndef XMALLOC_H
#define XMALLOC_H 1

static void *
xmalloc(size_t sz)
{
  void *p = malloc(sz);
  if (!p)
  {
    perror(NULL);
    exit(ENOMEM);
  }
  memset(p, 0, sz);
  return p;
}

#endif
