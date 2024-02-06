# starting at Jan. 2024
############################################################
# 1/8 -> 1/15 LinkedList
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        curr = head

        while curr:
            n = curr.next
            curr.next = prev
            prev = curr
            curr = n
        return prev

    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = head
        fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        seen = set()
        while headA:
            seen.add(headA)
            headA = headA.next
        while headB:
            if headB in seen:
                return headB
            headB = headB.next
        return None
    
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        seen = set()
        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        seen = set()
        while head:
            if head in seen:
                return head
            seen.add(head)
            head = head.next
        return None
    
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        prev = None
        curr = head
        while curr:
            n = curr.next
            curr.next = prev
            prev = curr
            curr = n
        return prev

    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        if left == right:
            return head
        dummy = ListNode(0)
        prev = dummy
        dummy.next = head
        for i in range(left-1):
            prev = prev.next
        head1 = prev
        tail = curr = prev.next
        step = (right-left)+1
        while step:
            step -= 1
            n = curr.next
            curr.next = prev
            prev = curr
            curr = n
        head1.next = prev
        tail.next = curr
        return dummy.next

    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        first_head = head
        if not head:
            return None
        if head.next:
            curr = second_head = second_head_dummy = head.next
        else:
            return head
        odd = True
        while curr.next:
            node = curr.next
            if odd:
                first_head.next = node
                first_head = node
            else:
                second_head.next = node
                second_head = node
            curr = node
            odd = not odd

        second_head.next = None
        first_head.next = second_head_dummy

        return dummy.next

############################################################
# 1/16 -> 1/21 
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        d = dict()
        for i in range(len(nums)):
            complementary = target - nums[i]
            if complementary in d:
                return [d[complementary], i]
            d[nums[i]] = i
    
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = set()
        nums1 = set(nums1)
        nums2 = set(nums2)
        for num in nums2:
            if num in nums1:
                res.add(num)
        return res

    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        d1 = dict()
        d2 = dict()
        for num in nums1:
            if num in d1:
                d1[num] += 1
            else:
                d1[num] = 1

        for num in nums2:
            if num in d2:
                d2[num] += 1
            else:
                d2[num] = 1

        for k, v in d1.items():
            if k in d2:
                v = min(d1[k], d2[k])
                res.extend([k] * v)
        return res

    def groupAnagrams(self, words):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        res = []
        d = collections.defaultdict(list)
        for word in words:
            order = [0] * 26
            for char in word:
                order[ord(char)-ord('a')] += 1
            d[tuple(order)].append(word)
        for v in d.values():
            res.append(v)
        return res
    
    def longestConsecutive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        s = set(nums)
        for num in nums:
            if num+1 in s:
                continue
            temp = 1
            while num-temp in s:
                temp+=1
            res = max(res, temp)
        return res
    
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        bulls = 0
        cows = 0
        match_idx = set()
        d = collections.defaultdict(set)
        for i in range(len(secret)):
            d[secret[i]].add(i)
        for i in range(len(guess)):
            if guess[i] == secret[i]:
                bulls += 1
                match_idx.add(i)
                d[guess[i]].remove(i)
                if not d[guess[i]]:
                    del d[guess[i]]
        for i in range(len(guess)):
            if i in match_idx:
                continue
            if guess[i] in d:
                cows += 1
                idx = d[guess[i]].pop()
                if not d[guess[i]]:
                    del d[guess[i]]
                
        return str(bulls)+"A"+str(cows)+"B"

    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        r_set = set()
        c_set = set()
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if matrix[r][c] == 0:
                    r_set.add(r)
                    c_set.add(c)

        
        for r in range(len(matrix)):
            for c in range(len(matrix[0])):
                if r in r_set or c in c_set:
                    matrix[r][c] = 0

        return matrix

    
############################################################
# 1/29 - 2/4
class TicTacToe(object):
    def __init__(self, n):
        """
        :type n: int
        """
        self.n = n
        self.r = [0] * n
        self.c = [0] * n
        self.dia = 0
        self.anti = 0

    def move(self, row, col, player):
        """
        :type row: int
        :type col: int
        :type player: int
        :rtype: int
        """
        token = 1 if player == 1 else -1
        self.r[row] += token 
        self.c[col] += token
        if row == col:
            self.dia += token
        if row == self.n-col-1:
            self.anti += token
        
        if self.r[row] == self.n or self.c[col] == self.n or self.dia == self.n or self.anti == self.n:
            return 1
        elif abs(self.r[row]) == self.n or abs(self.c[col]) == self.n or abs(self.dia) == self.n or abs(self.anti) == self.n:
            return 2
        return 0

class RandomizedSet(object):

    def __init__(self):
        self.arr = []
        self.map = dict()

    def insert(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if val not in self.map:
            self.map[val] = len(self.arr)
            self.arr.append(val)
            return True
        return False

    def remove(self, val):
        """
        :type val: int
        :rtype: bool
        """
        if val not in self.map:
            return False
        idx = self.map[val]
        last = self.arr[-1]        
        self.map[last] = idx
        del self.map[val]
        self.arr[idx] = last
        self.arr.pop()
        return True

    def getRandom(self):
        """
        :rtype: int
        """
        return random.choice(self.arr)

class Node(object):
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.next = None
        self.prev = None

class LRUCache(object):
    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.count = 0
        self.map = dict()
        self.head = Node(0, 0)
        self.tail = Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head
    def add_node(self, node):
        n = self.head.next
        self.head.next = node
        node.prev = self.head
        node.next = n
        n.prev = node

    def remove_node(self, node):
        n = node.next
        p = node.prev
        p.next = n
        n.prev = p

    def pop_tail(self):
        n = self.tail.prev
        self.remove_node(n)
        return n

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key not in self.map:
            return -1
        node = self.map[key]
        self.remove_node(node)
        self.add_node(node)
        return node.val

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key not in self.map:
            new_node = Node(key, value)
            self.count += 1
            self.map[key] = new_node
            self.add_node(new_node)
            if self.count > self.capacity:
                node = self.pop_tail()
                del self.map[node.key]
                self.count -= 1
        else:
            node = self.map[key]
            self.remove_node(node)
            node.val = value
            self.add_node(node)

############################################################
# 2/5 - 2/11
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for paren in s:
            if paren in ["(", "{", "["]:
                stack.append(paren)
            else:
                if paren == ")":
                    if stack and stack[-1] == "(":
                        stack.pop()
                    else:
                        return False
                elif paren == "}":
                    if stack and stack[-1] == "{":
                        stack.pop()
                    else:
                        return False
                else:
                    if stack and stack[-1] == "[":
                        stack.pop()
                    else:
                        return False
        return len(stack) == 0