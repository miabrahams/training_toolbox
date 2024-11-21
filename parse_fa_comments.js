function parseComments() {
  const commentContainers = document.querySelectorAll('.comment_container');
  const commentTree = [];
  let currentLevel = 0;
  let currentParent = null;

  commentContainers.forEach(container => {
    const width = parseFloat(container.style.width);
    const level = Math.round((100 - width) / 3);

    const comment = {
      container,
      level,
      children: []
    };

    if (level === 0) {
      // Top-level comment
      commentTree.push(comment);
      currentParent = comment;
      currentLevel = 0;
    } else if (level > currentLevel) {
      // Reply to the current parent comment
      currentParent.children.push(comment);
      currentParent = comment;
      currentLevel = level;
    } else if (level === currentLevel) {
      // Reply to the same parent as the previous comment
      currentParent.parent.children.push(comment);
      currentParent = comment;
    } else {
      // Reply to a higher-level parent
      while (currentLevel > level) {
        currentParent = currentParent.parent;
        currentLevel--;
      }
      currentParent.children.push(comment);
      currentParent = comment;
      currentLevel = level;
    }

    comment.parent = currentParent;
  });

  return commentTree;
}

let topComment = {
  container: document.querySelector('.journal-content-container'),
  level: -1,
  children: parseComments()
}

function grabCommentInfo(commentTree) {
  const container = commentTree.container;
  let comment = container.querySelector('.user-submitted-links')?.innerText;
  let author = container.querySelector('comment-username')?.innerText;
  let childComments = commentTree.children.map(grabCommentInfo)
  return {author, comment, childComments};
}

console.log(JSON.stringify(grabCommentInfo(topComment), null, 2))



