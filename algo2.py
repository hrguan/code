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
            if num-1 in s:
                continue
            temp = 1
            while num+temp in s:
                temp+=1
            res = max(res, temp)
        return res

############################################################