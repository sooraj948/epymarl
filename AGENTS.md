When creating implmentation documentation, always include the major design choices made, at the very top of the document so I, as a user, don't have to go through any code line-by-line.
Reduce the use of hyperparamters, where possible. The architecture itself should ideally accomodate for most scenarios and we can always finetune if needed.
When making major design changes that are worthy of documentation, always update the documentation after making the changes. It should be the last thing that you do. Add new walkthrough-{number}.md to the docs folder. (where {number} is one greater than the previous walkthrough).
Every 5 walkthroughs, compress all five into a single walkthrough doc. Don't lose any information, but keep it succinct.
Use git whenever major changes happen: commit the changes with a clear and descriptive commit message summarizing the major updates.
