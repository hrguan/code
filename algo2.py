# starting at Jan. 2024
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