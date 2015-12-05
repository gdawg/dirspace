# dirspace - recursively calculate usage by directory
---

dirspace is a utility which recursively calculates space by directory
with output similar to the following:

```bash
sh$ dirspace /System/Library --maxdepth 3
/System/Library/Accessibility,2309418,2meg
/System/Library/Accounts,4335924,4meg
/System/Library/Address Book Plug-Ins,3654925,3meg
/System/Library/Assistant,120670,117k
```

I know, you've already got du -skh, or find and a bunch of pipes - even still.

