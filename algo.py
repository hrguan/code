class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        res = 0
        left = 0
        right = len(height)-1
        while left < right:
            l = height[left]
            r = height[right]
            temp = min(l, r) * (right-left)
            if temp > res:
                res = temp
            if l < r:
                left += 1
            else:
                right -= 1
        return res
    
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        left = 0
        right = len(height)-1
        left_max = 0
        right_max = 0
        res = 0
        while left < right:
            if height[left] <= height[right]:
                #go left
                if height[left] > left_max:
                    left_max = height[left]
                else:
                    res += (left_max-height[left])
                left += 1
            else:
                #go right
                if height[right] > right_max:
                    right_max = height[right]
                else:
                    res += (right_max-height[right])
                right -= 1


        return res
    
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        # space
        # time
        if len(nums) < 3:
            return []
        res = set()
        nums.sort()
        for i in range(0, len(nums)-2):
            j = i+1
            k = len(nums)-1
            while j < k:
                if nums[i]+nums[j]+nums[k]==0:
                    res.add((nums[i], nums[j], nums[k]))
                    j += 1
                    k -= 1
                    
                elif nums[i]+nums[j]+nums[k] < 0:
                    j+=1
                else:
                    k-=1

        return res
    
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        # space
        # time
        res = float("inf")
        nums.sort()
        if len(nums) < 3:
            return 0
        for i in range(0, len(nums)-2):
            j = i+1
            k = len(nums)-1
            while j < k:
                temp = nums[i]+nums[j]+nums[k]
                if temp == target:
                    return target
                elif temp < target:
                    j+=1      
                else:
                    k-=1
                curr_diff = abs(temp-target)
                prev_diff = abs(res-target)

                if curr_diff < prev_diff:
                    res = temp
        return res
    
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = set()
        if len(nums) < 4:
            return []
        nums.sort()

        for a in range(0, len(nums)-3):
            for b in range(a+1, len(nums)-2):
                c = b+1
                d = len(nums)-1     
                while c < d:
                    temp = nums[a]+nums[b]+nums[c]+nums[d]
                    if temp == target:
                        res.add((nums[a], nums[b], nums[c], nums[d]))
                        c += 1
                        d -= 1
                    elif temp < target:
                        c += 1
                    else:
                        d -= 1
        return res
    
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        zero = 0
        one = 0
        two = len(nums)-1

        while one <= two:
            if nums[one] == 0:
                nums[one], nums[zero] = nums[zero], nums[one]
                zero += 1
                one += 1
            elif nums[one] == 1:
                one += 1
            else:
                nums[one], nums[two] = nums[two], nums[one]
                two -= 1

    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        left = 0
        right = len(s)-1
        
        while left <= right:
            l = ""
            r = ""
            while left <= len(s)-1 and not s[left].isalnum():
                left += 1
            if left <= len(s)-1:
                l = s[left].lower()
            while right >= 0 and not s[right].isalnum():
                right -= 1
            if right >= 0:
                r = s[right].lower()
            if l != r:
                return False
            left += 1
            right -=1
            

        return True
    
    def validPalindrome(self, s):
        left = 0
        right = len(s)-1

        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return self.check(s, left+1, right) or self.check(s, left, right-1)
        return True
    
    def check(self, s, left, right):
        while left <= right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return False
        return True
    
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        words = s.split()
        res = ""
        p = len(words)
        while p:
            if p == len(words):
                res += words[p-1]
            else:
                res += " " + words[p-1]
            p -= 1

        return res

    def reverseWords(self, s):
        """
        :type s: List[str]
        :rtype: None Do not return anything, modify s in-place instead.
        """
        left = 0
        right = len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        print(s)
        
        start = 0
        end = 0
        while end <= len(s)-1:
            while end <= len(s)-1 and s[end] != " ":
                end += 1
            e = end -1
            while start <= e:
                s[start], s[e] = s[e], s[start]
                start += 1
                e -= 1
            start = end+1
            end += 1
        
        ee = len(s)-1
        while start <= ee:
            s[start], s[ee] = s[ee], s[start]
            start += 1
            ee -= 1

    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = list(s)
        start = 0
        end = 0
        while end <= len(s)-1:
            while end <= len(s)-1 and s[end] != " ":
                end += 1
            e = end-1
            while start <= e:
                s[start], s[e] = s[e], s[start]
                start += 1
                e -= 1
            start = end+1
            end += 1
        
        e = len(s)-1
        while start <= e:
            s[start], s[e] = s[e], s[start]
            start += 1
            e -= 1
        
        return "".join(s)
    
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
    
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # left = 0
        # right = len(nums)-1

        # if nums[left] == nums[right]:
        #     return nums[left]
        # while left < right:
        #     middle = left+1
        #     while middle < right:
        #         if nums[middle] == nums[right] or nums[middle] == nums[left]:
        #             return nums[middle]
        #         middle += 1
        #     left += 1
        #     right -= 1

        s = nums[0]
        f = nums[0]

        while True:
            s = nums[s]
            f = nums[nums[f]]
            if s == f:
                break
        
        s = nums[0]

        while s != f:
            s = nums[s]
            f = nums[f]

        return s

    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        intervals.sort(key=lambda x:x[0])
        print(intervals)
        need = Queue.PriorityQueue()
        need.put(intervals[0][1])
        for i in range(1, len(intervals)):
            start = intervals[i][0]
            end = intervals[i][1]
            curr_end = need.get()
            if start < curr_end:
                need.put(curr_end)
            need.put(end)
        return need.qsize()

    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = head
        fast = head
        prev = None

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev = slow
        slow = slow.next
        prev.next = None

        while slow:
            n = slow.next
            slow.next = prev
            prev = slow
            slow = n
        
        slow = prev
        fast = head

        while slow:
            if fast.val != slow.val :
                return False
            fast = fast.next
            slow = slow.next
        return True
    
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: None Do not return anything, modify nums1 in-place instead.
        """
        last = m+n-1
        a = m-1
        b = n-1

        while b >= 0:
            if a >= 0 and nums1[a] > nums2[b]:
                nums1[last] = nums1[a]
                a -= 1
                
            else:
                nums1[last] = nums2[b]
                b -= 1
                
            last -= 1

    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        before_head = ListNode(0)
        before = before_head
        after_head = ListNode(0)
        after = after_head

        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next
        
        after.next = None
        before.next = after_head.next
        return before_head.next
    
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        curr = nums[0]
        idx = 1
        for i in range(1, len(nums)):
            if nums[i] == curr:
                continue
            nums[idx] = nums[i]
            curr = nums[i]
            idx +=1
        return idx
    
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        seen = set()
        while True:
            if n == 1:
                return True
            elif n in seen:
                return False
            else:
                seen.add(n)
                temp = 0
                for i in str(n):
                    temp += int(math.pow(int(i), 2))
                n = temp
