#ifndef __RBTREE_H
#define __RBTREE_H

#include <stdint.h>
#include <stddef.h>

#define RBT_OK                   0
#define RBT_NO_SUCH_KEY         -1
#define RBT_KEY_ALREADY_EXISTS  -2
#define RBT_NO_MEMORY           -4
#define RBT_IS_EMPTY            -8

typedef struct RBTree RBTree;
typedef struct RBNode RBNode;

enum Color {BLACK=0, RED=1};

struct RBTree {
    RBNode * root;
    size_t len;
    int (*compare)(const void *, const void *);
};

struct RBNode {
    enum Color color;
    RBNode * parent;
    RBNode * left;
    RBNode * right;
    const void * key;
};

/* Creates new red-black tree.
 * compare - pointer to comparison function.
*/
RBTree * rbtree_create(int (*compare)(const void *, const void *));

/* Frees memory allocated by RBTree object.
 */
void rbtree_free(RBTree * rbt);

/* Gets stored key.
 */
const void * rbtree_get(const RBTree * rbt, const void * key);

/* Adds new key to the tree. value can be NULL */
int rbtree_add(RBTree * rbt, const void * key);

/* Pops key from the tree. If key is NULL the
 * first available pair is popped.
 */
void * rbtree_pop(RBTree * rbt, const void * key);

/* Returns new array, that contains sorted keys.
 */
void * rbtree_to_array(const RBTree * rbt);

void rbtree_print(RBTree * rbt);

#endif
