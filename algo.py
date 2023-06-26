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

    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """

        #find peak
        for i in range(len(nums)-1, 0, -1):
            if nums[i - 1] < nums[i]:
                nums[i:] = sorted(nums[i:])

                pre_idx = i - 1
            
                for i in range(i, len(nums)):
                    if nums[pre_idx] < nums[i]:
                        nums[pre_idx], nums[i] = nums[i], nums[pre_idx]
                        return nums
        
        return nums.reverse()
    
    def circularArrayLoop(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        visit = set()
        for i in range(len(nums)):
            cycle = set()
            if i not in visit:
                while True:
                    if i in cycle:
                        return True
                    if i in visit:
                        break
                    visit.add(i)
                    cycle.add(i)
                    prev = i
                    i = (i+nums[i]) % len(nums)
                    if prev == i or ((nums[prev]>0) != (nums[i]>0)):
                        break
        return False
    
    def checkInclusion(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1) > len(s2):
            return False
        
        cnt1 = [0]*26
        cnt2 = [0]*26

        for i in range(len(s1)):
            cnt1[ord(s1[i])-ord('a')] += 1

        for i in range(len(s2)):
            cnt2[ord(s2[i])-ord('a')] += 1
            if i >= len(s1):
                cnt2[ord(s2[i-len(s1)])-ord('a')] -= 1

            if cnt1 == cnt2:
                return True

        return False

    def partitionLabels(self, s):
        """
        :type s: str
        :rtype: List[int]
        """
        res = []
        d = dict()
        for idx, char in enumerate(s):
            d[char] = idx
        
        start = 0
        end = 0
        for idx, char in enumerate(s):
            end = max(end, d[char])
            if end == idx:
                res.append(end-start+1)
                start = end+1
        
        return res
    
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0 
        buy = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < buy:
                buy = prices[i]
            else:
                res = max(res, prices[i]-buy)
        return res
    
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0
        buy = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < buy:
                buy = prices[i]
            else:
                res += (prices[i] - buy)
                buy = prices[i]

        return res
    
    def totalFruit(self, fruits):
        """
        :type fruits: List[int]
        :rtype: int
        """
        res = 0
        d = collections.defaultdict(int)

        l = 0
        for r, fruit in enumerate(fruits):
            d[fruit] += 1
            while len(d) > 2:
                d[fruits[l]] -= 1
                if d[fruits[l]] == 0:
                    del d[fruits[l]]
                l += 1
            res = max(res, r-l+1)

        return res
    
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        left = 0
        res = 0
        right = 0
        seen = dict()
        while right < len(s):
            if s[right] in seen:
                left = max(left, seen[s[right]]+1)
            res = max(res, right-left+1)
            seen[s[right]] = right
            right += 1
        return res
    
    def findRepeatedDnaSequences(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res, d = [], {}
        for i in range(len(s)):
            if s[i:i+10] not in d: 
                d[s[i :i+10]] = 0
            else:
                if s[i:i+10] not in res: 
                    res.append(s[i:i+10])
        
        return res
    
    def findClosestElements(self, arr, k, x):
        """
        :type arr: List[int]
        :type k: int
        :type x: int
        :rtype: List[int]
        """
        left = 0
        right = len(arr) - 1

        while right-left >= k:
            if abs(x-arr[left]) <= abs(x-arr[right]):  
                right -= 1
            else:
                left += 1
        return arr[left:right+1]
    
    def minSubArrayLen(self, target, nums):
        """
        :type target: int
        :type nums: List[int]
        :rtype: int
        """
        if sum(nums) < target:
            return 0
        elif sum(nums) == target:
            return len(nums)

        res = float("inf")
        left = 0
        temp = nums[left]
        if temp >= target:
            res = 1
        for right in range(1, len(nums)):   
            temp += nums[right]
            if temp >= target:
                res = min(res, right-left+1)
                while temp >= target:
                    temp -= nums[left]
                    left += 1
                    if temp >= target:
                        res = min(res, right-left+1)
        return res

    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        q = collections.deque()
        res = []
        for i in range(len(nums)):
            while q and nums[q[-1]] <= nums[i]:
                q.pop()
            q.append(i)
            if q[0] == i-k:
                q.popleft()
            if i >= k-1:
                res.append(nums[q[0]])
        return res
    
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """
        if len(t) > len(s):
            return ""
        
        left = 0
        start = 0
        end = 0
        missing = len(t)
        need = collections.Counter(t)
        for right, char in enumerate(s, 1):
            if need[char] > 0:
                missing -= 1
            need[char] -= 1
            if missing == 0:
                while left < right and need[s[left]] < 0:
                    need[s[left]] += 1
                    left += 1
                need[s[left]] += 1
                missing += 1
                if end == 0 or right -left < end-start:
                    start = left
                    end = right
                left+=1
        return s[start:end]
    
    def findLength(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: int
        """
        n2 = "".join([chr(x) for x in nums2])
        curr = ""
        res = 0
        for num in nums1:
            curr += chr(num)
            if curr in n2:
                res = max(res, len(curr))
            else:
                curr = curr[1:]
        return res

    def countSubarrays(self, nums, minK, maxK):
        """
        :type nums: List[int]
        :type minK: int
        :type maxK: int
        :rtype: int
        """
        currMin = -1
        currMax = -1
        currOver = -1
        res = 0
        for i in range(len(nums)):
            if nums[i] == minK:
                currMin = i
            if nums[i] == maxK:
                currMax = i
            if nums[i] > maxK or nums[i] < minK:
                currOver = i
            if currMin != -1 and currMax != -1:
                start = min(currMin, currMax)
                if currOver < start:
                    res += start - currOver
        return res
    
    def maxTurbulenceSize(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        res = 0
        temp = 0
        for i in range(len(arr)):
            if i >= 2 and (arr[i-2] > arr[i-1] < arr[i] or arr[i-2] < arr[i-1] > arr[i]):
                    temp += 1
            elif i >= 1:
                if arr[i] != arr[i-1]:
                    temp = 2
            else:
                temp = 1
            res = max(res, temp)
        return res

    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: bool
        """
        if not intervals:
            return True
        intervals.sort(key=lambda x:x[0])
        end = intervals[0][1]
        q = Queue.PriorityQueue()
        q.put(end)
        for i in range(1, len(intervals)):
            prev_end = q.get()
            curr_start = intervals[i][0]
            curr_end = intervals[i][1]
            if curr_start < prev_end:
                q.put(prev_end)
            q.put(curr_end)
        return True if q.qsize() == 1 else False
    
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        intervals.sort(key=lambda x:x[0])
        res = []
        res.append(intervals[0])
        for i in range(1, len(intervals)):
            end = res[-1][1]
            curr_start = intervals[i][0]
            curr_end = intervals[i][1]
            if curr_start <= end:
                res[-1][1] = max(curr_end, end)
            else:
                res.append(intervals[i])
                end = curr_end
        return res
    
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[List[int]]
        :type newInterval: List[int]
        :rtype: List[List[int]]
        """
        res = []
        for i in range(len(intervals)):
            if intervals[i][1] < newInterval[0]:
                res.append(intervals[i])
            elif intervals[i][0] > newInterval[1]:
                res.append(newInterval)
                newInterval = intervals[i]
            elif intervals[i][1] >= newInterval[0]:
                newInterval[0] = min(intervals[i][0], newInterval[0])
                newInterval[1] = max(intervals[i][1], newInterval[1])  
        res.append(newInterval)
        return res
    
    def intervalIntersection(self, firstList, secondList):
        """
        :type firstList: List[List[int]]
        :type secondList: List[List[int]]
        :rtype: List[List[int]]
        """
        i = 0
        j = 0
        res = []
        while i < len(firstList) and j < len(secondList):
            first_start = firstList[i][0]
            first_end = firstList[i][1]
            second_start = secondList[j][0]
            second_end = secondList[j][1]
            if first_start <= second_end and second_start <= first_end:
                res.append([max(first_start, second_start), min(first_end, second_end)])
            if first_end <= second_end:
                i += 1
            else:
                j += 1
        return res
    
    def employeeFreeTime(self, schedule):
        """
        :type schedule: [[Interval]]
        :rtype: [Interval]
        """
        intervals = []
        for employee in schedule:
            for interval in employee:
                intervals.append(interval)
        intervals.sort(key = lambda interval: interval.start)

        freeTime = []
        end = intervals[0].end
        for i in range(1, len(intervals)):
            start = intervals[i].start
            if start > end:
                freeTime.append(Interval(start = end, end = start))
                end = intervals[i].end
            else:
                end = max(intervals[i].end, end)
        return freeTime

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
    
    def swapNodes(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        first = head
        last = head
        for i in range(1, k):
            first = first.next
            
        null_checker = first 
        while null_checker.next:
            last = last.next
            null_checker = null_checker.next
        first.val, last.val = last.val, first.val
        return head

    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        jump = dummy
        dummy.next = head
        left = right = head

        while True:
            count = 0
            while right and count < k:
                count += 1
                right = right.next
            
            if count == k:
                prev = right
                curr = left
                for i in range(k):
                    nxt = curr.next
                    curr.next = prev
                    prev = curr
                    curr = nxt
                jump.next = prev
                jump = left
                left = right

            else:
                return dummy.next

    def reverseBetween(self, head, left, right):
        """
        :type head: ListNode
        :type left: int
        :type right: int
        :rtype: ListNode
        """
        dummy = ListNode(0)
        prev = dummy
        dummy.next = head
        for i in range(left-1):
            prev = prev.next
        curr = prev.next
        tail = None

        for i in range(right-left+1):
            nxt = curr.next
            curr.next = tail
            tail = curr
            curr = nxt
        
        prev.next.next = curr
        prev.next = tail

        return dummy.next
    
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        while head and head.next:
            tmp = head.next
            head.next = tmp.next
            tmp.next = head
            p.next = tmp
            
            head = head.next
            p = tmp.next
        return dummy.next
    
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        heap = []
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(heap, (lists[i].val, i))
        
        dummy = ListNode(0)
        curr = dummy
        while heap:
            val, i = heapq.heappop(heap)
            curr.next = lists[i]
            curr = curr.next
            if lists[i].next:
                lists[i] = lists[i].next
                heapq.heappush(heap, (lists[i].val, i))
        return dummy.next
    
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if not head:
            return None
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        prev = None
        curr = slow

        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        
        head_of_second_rev = prev
        
        first = head
        second = prev

        while second.next:
            nxt = first.next
            first.next = second
            first = nxt

            nxt = second.next
            second.next = first
            second = nxt

    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        def add(temp):
            curr = dummy
            while curr:
                if curr.val < temp.val:
                    prev = curr
                    curr = curr.next
                else:
                    break
            temp.next = curr
            prev.next = temp

        dummy = ListNode(-5001)
        while head:
            temp = head
            head = head.next
            temp.next = None
            add(temp)
        return dummy.next

    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        arr = []
        while head:
            arr.append(head.val)
            head = head.next
        
        arr.sort()
        dummy = ListNode(0)
        curr = dummy
        for i in range(len(arr)):
            curr.next = ListNode(arr[i])
            curr = curr.next
        return dummy.next

    def reverseEvenLengthGroups(self, head):
        """
        :type head: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        group = 2
        tail = head 
        while tail and tail.next:
            cnt = 1 
            cur = tail.next
            while cur.next and cnt < group:
                cur = cur.next
                cnt += 1
            pre, cur = tail, tail.next
            if cnt % 2 == 0: 
                while cnt and cur:
                    nxt = cur.next
                    cur.next = pre
                    pre = cur
                    cur = nxt
                    cnt -= 1
                first = tail.next
                first.next = cur
                tail.next = pre
                tail = first
            else:
                while cnt and cur:
                    pre, cur = cur, cur.next
                    cnt -= 1
                tail = pre
            group += 1
        return head

    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heap = []
        for num in nums:
            heapq.heappush(heap, -num)
        for i in range(k-1):
            heapq.heappop(heap)
        return heapq.heappop(heap)*-1
    
    def reorganizeString(self, s):
        """
        :type s: str
        :rtype: str
        """
        d = collections.Counter(s)

        heap = []
        for key, value in d.items():
            heapq.heappush(heap, (-value, key))
        
        max_count, max_char = heapq.heappop(heap)
        if max_count*-1 > (len(s)+1)//2:
            return ""
        
        idx = 0
        res = ["" for i in range(len(s))]
        
        for i in range((max_count*-1)):
            res[idx] = max_char
            idx += 2
        while heap:
            count, char = heapq.heappop(heap)
            for i in range(count*-1):
                if idx >= len(s):
                    idx = 1
                res[idx] = char
                idx += 2
        return ''.join(res)

    def kClosest(self, points, k):
        """
        :type points: List[List[int]]
        :type k: int
        :rtype: List[List[int]]
        """
        heap = []
        for x, y in points:
            temp = math.pow(x, 2) + math.pow(y, 2)
            distance = math.sqrt(temp)

            heapq.heappush(heap, (distance, [x, y]))
        res = []
        while k > 0:
            distance, point = heapq.heappop(heap)
            res.append(point)
            k -= 1
        return res
    
    def __init__(self, k, nums):
        """
        :type k: int
        :type nums: List[int]
        """
        self.k = k
        self.heap = nums[:k]
        heapq.heapify(self.heap)
        for i in range(k, len(nums)):
            heapq.heappush(self.heap, nums[i])
            heapq.heappop(self.heap)

    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap)> self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        count = collections.Counter(nums)
        heap = []
        for key, value in count.items():
            heapq.heappush(heap, (value, key))
            if len(heap)>k:
                heapq.heappop(heap)
        return [k for v, k in heap]
    
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        stack = []
        curr = root
        while True:
            while curr:
                stack.append(curr)
                curr = curr.left
            if not stack:
                break
            node = stack.pop()
            k -= 1
            if k == 0:
                return node.val
            curr = node.right
    
    def mostBooked(self, n, meetings):
        """
        :type n: int
        :type meetings: List[List[int]]
        :rtype: int
        """
        available = [r for r in range(n)]
        meeting_rooms = []
        used = [0]*n
        meetings.sort(key=lambda x:x[0])
        for start, end in meetings:
            while meeting_rooms and meeting_rooms[0][0] <= start:
                ending, room_no = heapq.heappop(meeting_rooms)
                heapq.heappush(available, room_no)
            if available:
                room_no = heapq.heappop(available)
                heapq.heappush(meeting_rooms, (end, room_no))
                used[room_no] += 1
            else:
                ending, room_no = heapq.heappop(meeting_rooms)
                heapq.heappush(meeting_rooms, (end+ending-start, room_no))
                used[room_no] += 1


        return used.index(max(used))

    def putMarbles(self, weights, k):
        """
        :type weights: List[int]
        :type k: int
        :rtype: int
        """
        if k == 1 or len(weights) == k:
            return 0
        m = sorted(weights[i]+weights[i-1] for i in range(1, len(weights)))

        return sum(m[-k+1:]) - sum(m[:k-1])

    def findCheapestPrice(self, n, flights, src, dst, k):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type k: int
        :rtype: int
        """
        adj = [[] for i in range(n)]
        for s, d, c in flights:
            adj[s].append((d, c))
        stop = 0
        q = collections.deque()
        q.append((src, 0))
        
        cost = [ float("inf") for i in range(n)]
        while q and stop <= k:
            for i in range(len(q)):
                curr, price = q.popleft()
                for nei, c in adj[curr]:
                    if price+c >= cost[nei]:
                        continue
                    cost[nei] = price+c
                    q.append((nei, price+c))
            stop += 1

        return -1 if cost[dst] == float("inf") else cost[dst]

    def minStoneSum(self, piles, k):
        """
        :type piles: List[int]
        :type k: int
        :rtype: int
        """
        heap = []
        for pile in piles:
            heapq.heappush(heap, -pile)
        for i in range(k):
            curr = heapq.heappop(heap) * -1
            remove = curr // 2
            curr -= remove
            heapq.heappush(heap, -curr)
        return sum(heap) * -1
    
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        if n == 0:
            return len(tasks)
        
        count = collections.Counter(tasks)
        freq = count.values()
        max_count = max(freq)
        last = freq.count(max_count)

        ans = (max_count-1) * (n+1) + last
        print(ans, len(tasks))
        return max(ans, len(tasks))
    
        # if n == 0:
        #     return len(tasks)
        # counter = collections.Counter(tasks)
        # res = 0
        # n += 1
        # while counter:
        #     ready = counter.most_common(n)
        #     res += len(ready)
        #     for k, v in ready:
        #         if counter[k] > 1:
        #             counter[k] -= 1
        #         else:
        #             del counter[k]
        #     if counter:
        #         res += (n-len(ready))
        # return res

    def __init__(self):
        self.low = []
        self.high = []
    def addNum(self, num):
        """
        :type num: int
        :rtype: None
        """
        if len(self.low) == len(self.high):
            heapq.heappush(self.low, -num)
            n = heapq.heappop(self.low)
            heapq.heappush(self.high, -n)

        else:
            heapq.heappush(self.high, num)
            n = heapq.heappop(self.high)
            heapq.heappush(self.low, -n)
    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.low) == len(self.high):
            low = self.low[0] * -1
            ans = (self.high[0] + low )/2.0
            return ans
        else:
            return float(self.high[0])
        
    def medianSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[float]
        """
        from sortedcontainers import SortedList
        res = []
        sl = SortedList()
        for i in range(len(nums)):
            sl.add(nums[i])
            if i < k-1:
                continue
            if k % 2 != 0:
                res.append(sl[k//2]*1.0)
            else:
                s = (sl[k//2] + sl[k//2-1]) / 2.0
                res.append(s)
            sl.remove(nums[i-k+1])
        return res

    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        l = matrix[0][0]
        r = matrix[-1][-1]

        def less_k(m):
            res = 0
            for i in range(len(matrix)):
                cnt = bisect_right(matrix[i], m)
                res += cnt
            return res

        while l < r:
            mid = (l+r) // 2
            if less_k(mid) < k:
                l = mid+1
            else:
                r = mid
        return l

    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        heap = []
        for i in range(min(k, len(nums1))):
            heapq.heappush(heap, (nums1[i]+nums2[0], nums1[i], nums2[0], 0))
        
        res = []
        while k > 0 and heap:
            sum, n1, n2, idx = heapq.heappop(heap)
            res.append([n1, n2])
            if idx + 1 < len(nums2):
                heapq.heappush(heap, (n1+nums2[idx + 1], n1, nums2[idx + 1], idx + 1))
            k -= 1
        return res

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        total = len(nums1)+len(nums2)
        half = total // 2
        if len(nums2) < len(nums1):
            nums1, nums2 = nums2, nums1
        l = 0
        r = len(nums1)-1
        while True:
            i = (l+r) //2
            j = half-i-2
            Aleft = nums1[i] if i >= 0 else float("-inf")
            Aright = nums1[i+1] if i+1 < len(nums1) else float("inf")
            Bleft = nums2[j] if j >= 0 else float("-inf")
            Bright = nums2[j+1] if j+1 < len(nums2) else float("inf")

            if Aleft <= Bright and Bleft <= Aright:
                if total%2 == 0:
                    return (max(Aleft, Bleft)+min(Aright, Bright)) /2.0
                else:
                    return min(Aright, Bright) * 1.0
            elif Aleft >Bright:
                r = i-1
            else:
                l = i+1

    def subarraysWithKDistinct(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        def atMost(nums, k):
            res = 0
            d = dict()
            left = 0
            right = 0
            for i in range(len(nums)):
                if nums[i] not in d:
                    d[nums[i]] = 1
                else:
                    d[nums[i]] += 1
                
                while len(d) > k:
                    d[nums[left]] -= 1
                    if d[nums[left]] == 0:
                        del d[nums[left]]
                    left += 1
                
                res += right-left+1
                right +=1
                
            return res

        return atMost(nums, k) - atMost(nums, k-1)

    def sortArray(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        heap = []
        res = []
        for num in nums:
            heapq.heappush(heap, num)
        while heap:
            res.append(heapq.heappop(heap))
        return res

    def getOrder(self, tasks):
        """
        :type tasks: List[List[int]]
        :rtype: List[int]
        """
        res = []
        heap = []
        tasks = sorted((task, i) for i, task in enumerate(tasks))
        i = 0
        curr = 0
        
        for (e, p), i in tasks:
            while heap and curr < e:
                proc, idx, en = heapq.heappop(heap)
                res.append(idx)
                curr = max(curr, en) + proc
            heapq.heappush(heap, (p, i, e))
        while heap:
            proc, idx, en = heapq.heappop(heap)
            res.append(idx)

        return res

    def minKBitFlips(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        flip_count = 0
        flip_queue = collections.deque()
        for i,num in enumerate(nums):
            condition1 = num==0 and len(flip_queue)%2==0
            condition2 = num==1 and len(flip_queue)%2!=0
            if condition1 or condition2:
                flip_count += 1
                flip_queue.append(i+k-1)
            if flip_queue and i >= flip_queue[0]:
                flip_queue.popleft()
        return -1 if flip_queue else flip_count

    def minWindow(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: str
        """
        if len(s2) > len(s1):
            return ""
        def find_end(start1):
            start2 = 0
            while start1 < len(s1):
                if s1[start1] == s2[start2]:
                    start2+=1
                    if start2 == len(s2):
                        break
                start1+=1
            return start1 if start2 == len(s2) else None
        def better_start(end1):
            end2 = len(s2)-1
            while end2 >= 0:
                if s1[end1] == s2[end2]:
                    end2 -=1
                end1 -=1
            return end1+1

        start1 = 0
        res = ""
        length = float("inf")
        while start1 < len(s1):
            end = find_end(start1)
            if end == None:
                break
            b_start = better_start(end)
            if end-b_start+1 < length:
                length = end-b_start+1
                res = s1[b_start:end+1]
            start1 = b_start+1
        return res

    def minimumDeviation(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        heap = []
        for num in nums:
            if num % 2 !=0:
                heap.append((num, num*2))
            else:
                temp = num
                while temp%2==0:
                    temp //=2
                heap.append((temp, max(temp*2, num))) 
        max_num = max([i for i, j in heap])
        heapq.heapify(heap)

        res = float("inf")
        while len(heap) == len(nums):
            MIN, MAX = heapq.heappop(heap)
            if max_num - MIN < res:
                res = max_num - MIN
            if MIN < MAX:
                heapq.heappush(heap, (MIN*2, MAX))
                max_num = max(max_num, MIN*2)
        return res

    def findMaximizedCapital(self, k, w, profits, capital):
        """
        :type k: int
        :type w: int
        :type profits: List[int]
        :type capital: List[int]
        :rtype: int
        """
        projects = [(capital[i], profits[i]) for i in range(len(profits))]
        projects.sort()
        heap = []
        for _ in range(k):
            while projects and projects[0][0] <= w:
                capital, profit = projects.pop(0)
                heapq.heappush(heap, -profit)
            if not heap:
                break
            w -= heapq.heappop(heap)
        return w
    
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        left = 1
        right = n
        while left < right:
            mid = (left+right)//2
            if isBadVersion(mid) == False:
                left=mid+1
            else:
                right=mid
        return left
    
    def singleNonDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        left = 0
        right = len(nums)-1
        if len(nums) == 1:
            return nums[0]
        while left<=right:
            mid = (left+right)//2
            if (mid+1 <len(nums) and nums[mid]!=nums[mid+1]) and (mid-1 >= 0 and nums[mid]!=nums[mid-1]):
                return nums[mid]
            a = mid%2==0 and nums[mid]==nums[mid-1]
            b = mid%2!=0 and nums[mid]==nums[mid+1]
            if a or b:
                right = mid-1
            else:
                left = mid+1
        return nums[right]

    def __init__(self, w):
        """
        :type w: List[int]
        """
        self.accu = []
        self.total = 0
        for weight in w:
            self.total+=weight
            self.accu.append(self.total)

    def pickIndex(self):
        """
        :rtype: int
        """
        a = random.randint(1, self.total)
        return bisect.bisect_left(self.accu, a)
    
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums)-1
        while left<=right:
            mid = (left+right)//2
            if nums[mid] == target:
                return mid
            if nums[left]<=nums[mid]: 
                if nums[left]<=target<nums[mid]:
                    right = mid-1
                else:
                    left = mid+1
            else:
                if nums[mid]<target<=nums[right]:
                    left = mid+1
                else:
                    right = mid-1

        return -1
    
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        left = 0
        right = len(nums)-1
        while left<=right:
            mid = (left+right)//2
            print(left, mid, right)
            if nums[mid] == target or nums[left] == target or nums[right] == target:
                return True
            if nums[left]<=nums[mid]: 
                if nums[left]<=target<=nums[mid]:
                    right = mid-1
                else:
                    left += 1
            else:
                if nums[mid]<=target<=nums[right]:
                    left = mid+1
                else:
                    right -= 1

        return False
    
    def numRescueBoats(self, people, limit):
        """
        :type people: List[int]
        :type limit: int
        :rtype: int
        """
        people.sort()
        low = 0
        high = len(people)-1
        res = 0
        while low<=high:
            if people[low]+people[high] <= limit:
                res += 1
                low += 1
                high -= 1
            else:
                res += 1
                high -= 1
        return res
    
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        trip = 0
        curr = 0
        start = 0
        for i in range(len(gas)):
            trip += gas[i] - cost[i]
            curr += gas[i] - cost[i]
            if curr < 0:
                curr = 0
                start = i+1
        if start<len(gas) and trip >= 0:
            return start
        return -1
    
    def minRefuelStops(self, target, startFuel, stations):
        """
        :type target: int
        :type startFuel: int
        :type stations: List[List[int]]
        :rtype: int
        """
        stations.append([target, 0])
        res = 0
        prev = 0
        currfuel = startFuel
        available = []
        for position, fuel in stations:
            distance = position - prev
            prev = position
            if currfuel < distance:
                while available and currfuel < distance:
                    gas = heapq.heappop(available)*-1
                    currfuel += gas
                    res += 1
                if currfuel < distance:
                    return -1

            currfuel -= distance
            heapq.heappush(available, -fuel)
        return res

    def twoCitySchedCost(self, costs):
        """
        :type costs: List[List[int]]
        :rtype: int
        """
        SUM_B = sum(j for i, j in costs)
        diff = sorted(i-j for i, j in costs)
        save = sum(diff[:len(costs)//2])
        return SUM_B+save

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 1:
            return 1
        if n == 2:
            return 2
        
        n1 = 1
        n2 = 2
        res = 0
        for i in range(3, n+1):
            res = n1+n2
            n1, n2 = n2, res
            
        return res

    def tribonacci(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n == 0:
            return 0
        if n <= 2:
            return 1
        n0 = 0
        n1 = 1
        n2 = 1
        res = 0
        for i in range(3, n+1):
            res = n0+n1+n2
            n0, n1, n2 = n1, n2, res
        return res
    
    def canPartition(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        total = sum(nums)
        if total % 2 != 0:
            return False
        half = total //2
        dp = [False] * (half+1)
        dp[0] = True
        for num in nums:
            for i in range(half, num-1, -1):
                dp[i] = dp[i] or dp[i-num]
        return dp[half]

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        dp = [float("inf")] * (amount+1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount+1):
                dp[i] = min(dp[i], dp[i-coin]+1)
        return -1 if dp[amount] == float("inf") else dp[amount]

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        last = len(nums)-1
        for i in range(len(nums)-2, -1, -1):
            if nums[i]+i >= last:
                last = i
        return last == 0
    
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [0]*len(nums)
        j = 0
        for i in range(1, len(nums)):
            while j < len(nums) and j+nums[j] < i:
                j+=1
            dp[i] = dp[j]+1 
        return dp[-1]

    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        start = 0
        nums.sort()
        for num in nums:
            if num == start:
                start += 1
            else:
                return start
        return start
    
    def findKthPositive(self, arr, k):
        """
        :type arr: List[int]
        :type k: int
        :rtype: int
        """
        l = 0
        r = len(arr)-1
        while l <= r:
            m = (l+r)//2
            if arr[m]-m-1 < k:
                l = m+1
            else:
                r = m-1
        return l+k

    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        for i in range(n):
            while 1 <= nums[i]<=n and nums[nums[i]-1] != nums[i]:
                nums[nums[i]-1], nums[i] = nums[i], nums[nums[i]-1]
        for i in range(n):
            if nums[i] != i+1:
                return i+1
        return n+1
    
    def maximumGap(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return 0
        nums.sort()
        gap = 0
        for i in range(1, len(nums)):
            if nums[i] - nums[i-1] > gap:
                gap = nums[i] - nums[i-1]
        return gap
    
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        def comparator(x1, x2):
            if x1+x2 > x2+x1:
                return 1
            elif x1+x2 < x2+x1:
                return -1
            return 0
        
        nums = [str(num) for num in nums]
        nums.sort(key=functools.cmp_to_key(comparator), reverse=True)
        return "0" if nums[0] == "0" else "".join(nums)
    
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        n = sorted(nums, reverse=True)
        for i in range(1, len(nums), 2):
            nums[i] = n.pop(0)
        for i in range(0, len(nums), 2):
            nums[i] = n.pop(0)

    def rearrangeBarcodes(self, barcodes):
        """
        :type barcodes: List[int]
        :rtype: List[int]
        """
        d = collections.Counter(barcodes)
        heap = []
        for k, v in d.items():
            heapq.heappush(heap, (-v, k))
        res = [None]*len(barcodes)
        i = 0
        while heap:
            count, key = heapq.heappop(heap)
            while count != 0:
                if i >= len(res):
                    i = 1
                res[i] = key
                i += 2
                count += 1

        return res

    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for paren in s:
            if paren == "(" or paren == "[" or paren == "{":
                stack.append(paren)
            elif paren == ")":
                if stack and stack[-1] == "(":
                    stack.pop()
                else:
                    return False
            elif paren == "]": 
                if stack and stack[-1] == "[":
                    stack.pop()
                else:
                    return False
            elif paren == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
                else:
                    return False
        
        return len(stack) == 0

    def removeDuplicates(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        stack.append(s[0])
        for i in range(1, len(s)):
            if stack and stack[-1] == s[i]:
                stack.pop()
            else:
                stack.append(s[i])
        return ''.join(stack)

    def minRemoveToMakeValid(self, s):
        """
        :type s: str
        :rtype: str
        """
        stack = []
        ss = list(s)
        for i in range(len(ss)):
            if ss[i] == "(":
                stack.append(i)
            elif ss[i] == ")":
                if stack:
                    stack.pop()
                else:
                    ss[i] = ""
        if stack:
            for idx in stack:
                ss[idx] = ""


        return "".join(ss)

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        preCourse_canTake = collections.defaultdict(list)
        degree = [0] * numCourses
        for course, pre in prerequisites:
            preCourse_canTake[pre].append(course)
            degree[course] += 1
        
        queue = deque()
        for i in range(len(degree)):
            if degree[i] == 0:
                queue.append(i)
        while queue:
            curr = queue.popleft()
            for canTake in preCourse_canTake[curr]:
                degree[canTake] -= 1
                if degree[canTake] == 0:
                    queue.append(canTake)
    
        for i in range(len(degree)):
            if degree[i] != 0:
                return False
        return True
    
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        preCourse_canTake = collections.defaultdict(list)
        degree = [0] * numCourses
        for course, pre in prerequisites:
            preCourse_canTake[pre].append(course)
            degree[course] += 1
        
        queue = deque()
        for i in range(len(degree)):
            if degree[i] == 0:
                queue.append(i)
        res = []
        while queue:
            curr = queue.popleft()
            res.append(curr)
            for canTake in preCourse_canTake[curr]:
                degree[canTake] -= 1
                if degree[canTake] == 0:
                    queue.append(canTake)
                
        return res if len(res) == numCourses else []

##################################
    def num_perfect_pairs(arr):
        arr.sort(key=lambda x:abs(x))
        left = 0
        right = 1
        cnt = 0
        while left < len(arr)-1 and right < len(arr):
            x = abs(arr[left])
            y = abs(arr[right])
            if y <= 2*x:
                cnt += right-left
                right += 1
            else:
                left += 1
        return cnt

    def job_execution(n, executionTime, x, y):
        executionTime.sort(reverse=True)
        i = 0
        cnt = 0
        while True:
            if executionTime[i] > 0:
                executionTime[i] -= x
                for j in range(1, len(executionTime)):
                    executionTime[j] -= y
                executionTime.sort(reverse=True)
                cnt += 1
            else:
                return cnt
            
    def string_formation(self, words, target):
        """
        :type words: List[str]
        :type target: str
        :rtype: int
        """
        dp=[0]*(len(target)+1)
        dp[0]=1
        count=[[0]*26 for _ in range(len(words[0]))]
        for i in range(len(words[0])):
            for word in words:
                count[i][ord(word[i])-ord('a')]+=1
        for i in range(len(words[0])):
            for j in range(len(target)-1,-1,-1):
                dp[j+1]+= dp[j]*count[i][ord(target[j])-ord('a')]
                dp[j+1]%= (10**9+7)

        return dp[-1] 

    def max_array_value(a):
        i, j = min(a), max(a)
        while i < j:
            m = (i + j) // 2
            avail = 0
            canMake = True
            for el in a:
                if el <= m:
                    avail += m - el
                else:
                    mustSubtract = max(0, el - m)
                    if mustSubtract > avail:
                        canMake = False
                        break
                    avail -= mustSubtract
            if canMake:
                j = m
            else:
                i = m + 1
        return i
    
    def task_scheduling(n, c, t):
        def f(i, j, memo):
            if i == n:
                return [float('inf'), 0][j >= 0]
            if (i, j) in memo:
                return memo[(i, j)]
        
            result = min(c[i] + f(i + 1, j + t[i], memo), f(i + 1, j - 1, memo))
            memo[(i, j)] = result
            return result
        memo = {}
        return f(0, 0, memo)
    
    def cross_the_threshold(n, initEnergy, threshold):
        res = 0
        s = sum(initEnergy)
        while s > threshold:
            res += 1
            temp = 0
            for i in range(n):
                initEnergy[i] -= 1
                if initEnergy[i] > 0:
                    temp += initEnergy[i]
            s = temp
        return res -1

    #https://www.geeksforgeeks.org/number-of-distinct-words-of-size-n-with-at-most-k-contiguous-vowels/
    def power(x, y, p):
        result = 1
        x = x % p
        if x == 0:
            return 0
        while y > 0:
            if y % 2 == 1:
                result = (result * x) % p
            y = y // 2
            x = (x * x) % p
        return result

    def power(x, y, p):
        res = 1
        x = x % p
        if (x == 0):
            return 0
        while (y > 0):
            if (y & 1):
                res = (res * x) % p
            y = y >> 1
            x = (x * x) % p
        return res
 
    def kvowelwords(N, K):
        i, j = 0, 0
        MOD = 1000000007
        dp = [[0 for i in range(K + 1)] for i in range(N + 1)]
        sum = 1
        for i in range(1, N + 1):
            dp[i][0] = sum * 21
            dp[i][0] %= MOD
            sum = dp[i][0]
            for j in range(1, K + 1):
                if (j > i):
                    dp[i][j] = 0
                elif (j == i):
                    dp[i][j] = power(5, i, MOD)
                    #dp[i][j] = (5 ** i) % MOD
                else:
                    dp[i][j] = dp[i - 1][j - 1] * 5
                dp[i][j] %= MOD
                sum += dp[i][j]
                sum %= MOD
    
        return sum

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def dfs(nums, index, path, res):
            res.append(path)
            for i in range(index, len(nums)):
                dfs(nums, i+1, path+[nums[i]], res)
        res = []
        dfs(sorted(nums), 0, [], res)
        return res
    
        def backtrack(start, end, tmp):
            res.append(tmp[:])
            for i in range(start, end):
                tmp.append(nums[i])
                backtrack(i+1, len(nums), tmp)
                tmp.pop()
                
        res = []
        backtrack(0, len(nums), [])
        return res

    def subsetsWithDup(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(start, end, tmp):
            res.append(tmp[:])
            for i in range(start, end):
                if i != start and nums[i] == nums[i-1]:
                    continue
                tmp.append(nums[i])
                backtrack(i+1, len(nums), tmp)
                tmp.pop()

        nums.sort()     
        res = []
        backtrack(0, len(nums), [])
        return res

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        phone = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        res = []
        def backtrack(temp, digits):
            if not digits:
                return res.append(temp)
            for letter in phone[digits[0]]:
                backtrack(temp+letter, digits[1:])
        backtrack("", digits)
        return res
    
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def backtrack(open, close, temp):
            if open == close == n:
                return res.append("".join(temp))
            if open < n:
                temp.append("(")
                backtrack(open+1, close, temp)
                temp.pop()
            if close < n and open>close:
                temp.append(")")
                backtrack(open, close+1, temp)
                temp.pop()

        res = []
        backtrack(0, 0, [])
        return res

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def backtrack(nums, temp, res):
            if not nums:
                res.append(temp[:])
            
            for i in range(len(nums)):
                newNums = nums[:i] + nums[i+1:]
                temp.append(nums[i])
                backtrack(newNums, temp, res)
                temp.pop()

        res = []
        backtrack(nums, [], res)
        return res
    
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        def backtrack(remain, temp, idx):
            if remain == 0:
                res.append(temp[:])
            for i in range(idx, n+1):
                temp.append(i)
                backtrack(remain-1, temp, i+1)
                temp.pop()
        res = []
        backtrack(k, [], 1)
        return res
    
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def backtrack(idx, temp, target):
            if target < 0:
                return 
            if target == 0:
                return res.append(temp[:])
            for i in range(idx, len(candidates)):
                temp.append(candidates[i])
                backtrack(i, temp, target-candidates[i])
                temp.pop()
        res = []
        backtrack(0, [], target)
        return res

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def backtrack(idx, temp, target):
            if target < 0:
                return 
            if target == 0:
                return res.append(temp[:])
            for i in range(idx, len(candidates)):
                if i!=idx and candidates[i] == candidates[i-1]:
                    continue
                temp.append(candidates[i])
                backtrack(i+1, temp, target-candidates[i])
                temp.pop()
        res = []
        candidates.sort()
        backtrack(0, [], target)
        return res
 
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        # if not root:
        #     return 0
        # return max(self.maxDepth(root.left), self.maxDepth(root.right))+1
    
        if not root:
            return 0
        level = 0
        level_nodes = 1
        q = deque()
        q.append(root)
        while q:
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            level_nodes -= 1
            if level_nodes == 0:
                level += 1
                level_nodes = len(q)
        return level
    
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.right)
        self.invertTree(root.left)
        
        return root

    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.ans = 0
        def dfs(node):
            if not node:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)
            self.ans = max(self.ans, left+right)
            return max(left, right) + 1
        dfs(root)
        return self.ans

    def __init__(self):
        self.prev = None
    def flatten(self, root):
        """
        :type root: TreeNode
        :rtype: None Do not return anything, modify root in-place instead.
        """
        if not root:
            return None
        self.flatten(root.right)
        self.flatten(root.left)
        root.left = None
        root.right = self.prev
        self.prev = root

    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        self.res = float("-inf")
        def dfs(node):
            if not node:
                return 0
            left = max(dfs(node.left), 0)
            right = max(dfs(node.right), 0)
            self.res = max(self.res, left+right+node.val)
            return node.val + max(left, right)
        dfs(root)
        return self.res

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        s = []
        def dfs(root):
            if not root:
                return s.append('n')
            s.append(str(root.val))
            dfs(root.left)
            dfs(root.right)
        dfs(root)
        return ' '.join(s)      

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        q = deque(data.split())
        def dfs():
            node = q.popleft()
            if node == 'n':
                return None
            root = TreeNode(node)
            root.left = dfs()
            root.right = dfs()
            return root
        return dfs()
    
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def bfs(k):
            if len(k) == 0:
                return
            temp = []
            new = []
            for node in k:
                temp.append(node.val)
                if node.left:
                    new.append(node.left)
                if node.right:
                    new.append(node.right)
            res.append(temp)
            bfs(new)
        res = []
        if root:
            bfs([root])
        return res
    
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        def bfs(k, flag):
            if len(k) == 0:
                return
            temp = []
            new = []
            for node in k:
                temp.append(node.val)
                if node.left:
                    new.append(node.left)
                if node.right:
                    new.append(node.right)
            if flag == 1:    
                res.append(temp)
            else:
                res.append(temp[::-1])
            flag *= -1
            bfs(new, flag)
        res = []
        flag = 1
        if root:
            bfs([root], flag)
        return res

    def connect(self, root):
        """
        :type root: Node
        :rtype: Node
        """
        if not root:
            return
        q = [root]
        while q:
            curr = q.pop(0)
            if curr.left and curr.right:
                curr.left.next = curr.right
                if curr.next:
                    curr.right.next = curr.next.left
                q.append(curr.left)
                q.append(curr.right)
        return root

    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        stack = [root]
        parent = {root: None}
        while not(p in parent and q in parent):
            node = stack.pop()
            if node.left:
                stack.append(node.left)
                parent[node.left] = node
            if node.right:
                stack.append(node.right)
                parent[node.right] = node
        common = set()
        while p:
            common.add(p)
            p = parent[p]
        while q not in common:
            q = parent[q]
        return q
    
    def verticalTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        dic = collections.defaultdict(list)
        
        def helper(placement, level, node):
            if not node:
                return
            dic[placement].append((level, node.val))
            helper(placement-1, level+1, node.left)
            helper(placement+1, level+1, node.right)
        helper(0, 0, root)
        res = []
        for i in sorted(dic.keys()):
            temp = []
            for j in sorted(dic[i]):
                temp.append(j[1])
            res.append(temp)
        return res

    def __init__(self):
        self.data = [None]*1000001

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        self.data[key] = value

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        v = self.data[key]
        return v if v != None else -1

    def remove(self, key):
        """
        :type key: int
        :rtype: None
        """
        self.data[key] = None
    
    def __init__(self):
        self.dic = {}
    def shouldPrintMessage(self, timestamp, message):
        """
        :type timestamp: int
        :type message: str
        :rtype: bool
        """
        if message not in self.dic:
            self.dic[message] = timestamp+9
            return True
        else:
            val = self.dic[message]
            if timestamp > val:
                self.dic[message] = timestamp+9
                return True
            else:
                return False
    
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        res = []
        stack = []
        d = {}
        for num in nums2:
            while stack and num > stack[-1]:
                d[stack.pop()] = num
            stack.append(num)
        while stack:
            d[stack.pop()] = -1
        for num in nums1:
            res.append(d[num])
        return res
    
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        s1 = []
        s2 = []
        for n in s:
            s1.append(s.index(n))
        for n in t:
            s2.append(t.index(n))
        print(s1, s2)
        if s1 == s2:
            return True
        return False

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: int
        """
        odd = set()
        for letter in s:
            if letter not in odd:
                odd.add(letter)
            else:
                odd.remove(letter)
        if len(odd) == 0:
            return len(s)
        else:
            return len(s) - len(odd) + 1
        
    def fractionToDecimal(self, numerator, denominator):

        n, remainder = divmod(abs(numerator), abs(denominator))
        sign = '-' if numerator*denominator < 0 else ''
        result = [sign+str(n), '.']
        remainders = {}

        while remainder > 0 and remainder not in remainders:
            remainders[remainder] = len(result)
            n, remainder = divmod(remainder*10, abs(denominator))
            result.append(str(n))

        if remainder in remainders:
            idx = remainders[remainder]
            result.insert(idx, '(')
            result.append(')')

        return ''.join(result).rstrip(".")

    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        d = collections.defaultdict(int)
        for letter in magazine:
            if letter not in magazine:
                d[letter] = 1
            else:
                d[letter] += 1
        for word in ransomNote:
            if d[word] <= 0:
              return False
            d[word] -= 1
        return True

    def canPermutePalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        odd = True
        d = {}
        for letter in s:
            if letter not in d:
                d[letter] = 1
            else:
                d[letter] += 1
        
        for k, v in d.items():
            if v % 2 == 0:
                continue
            elif v % 2 != 0 and odd == True:
                odd = False
            else:
                return False
        return True

    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        d = {}
        for letter in s:
            if letter in d:
                d[letter] += 1
            else:
                d[letter] = 1
        for idx in range(len(s)):
            if d[s[idx]] == 1:
                return idx
        return -1

    def groupAnagrams(self, words):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        d = collections.defaultdict(list)
        for word in words:
            key = [0]*26
            for letter in word:
                key[ord(letter)-ord('a')] += 1
            d[tuple(key)].append(word)
        res = []
        for v in d.values():
            res.append(v)
        return res

    def __init__(self, n):
        """
        :type n: int
        """
        self.n = n
        self.row = [0] * n
        self.col = [0] * n
        self.dia = 0
        self.anti = 0

    def move(self, row, col, player):
        """
        :type row: int
        :type col: int
        :type player: int
        :rtype: int
        """
        val = 1 if player == 1 else -1

        self.row[row] += val
        self.col[col] += val
        if row == col:
            self.dia += val
        if row == self.n-1-col:
            self.anti += val
        if self.row[row] == self.n or self.col[col] == self.n or self.dia == self.n or self.anti == self.n:
            return 1
        if abs(self.row[row]) == self.n or abs(self.col[col]) == self.n or abs(self.dia) == self.n or abs(self.anti) == self.n:
            return 2

        return 0
    
    def __init__(self):
        self.count = collections.defaultdict(int)
        self.idx = 0
        self.heap = []  
    def push(self, val):
        """
        :type val: int
        :rtype: None
        """
        self.count[val] += 1
        self.idx += 1
        heappush(self.heap, (-self.count[val], -self.idx, val))
    def pop(self):
        """
        :rtype: int
        """
        _, _, val = heappop(self.heap)
        self.count[val] -=1
        return val
    
    def findAnagrams(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: List[int]
        """
        p_counter = Counter(p)
        s_counter = dict()
        res = []
        for i in range(len(s)-len(p)+1):
            if i == 0:
                s_counter = Counter(s[:len(p)])
            else:
                s_counter[s[i-1]] -= 1
                s_counter[s[i+len(p)-1]] += 1
            if len(p_counter-s_counter) == 0:
                res.append(i)
        return res

    def mergeTwoLists(self, list1, list2):
        """
        :type list1: Optional[ListNode]
        :type list2: Optional[ListNode]
        :rtype: Optional[ListNode]
        """
        l1 = list1
        l2 = list2
        dummy = ListNode(0)
        curr = dummy
        while l1!=None and l2!=None:
            if l1 and l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
                curr = curr.next
            else:
                curr.next = l2
                l2 = l2.next
                curr = curr.next
        if l1:
            curr.next = l1
        else:
            curr.next = l2
        return dummy.next

    class Node(object):
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None
class BrowserHistory(object):
    def __init__(self, homepage):
        """
        :type homepage: str
        """      
        self.root = Node(homepage)
    def visit(self, url):
        """
        :type url: str
        :rtype: None
        """
        new = Node(url)
        self.root.next = new
        new.prev = self.root
        self.root = new
    def back(self, steps):
        """
        :type steps: int
        :rtype: str
        """
        while steps and self.root.prev:
            steps -= 1
            self.root = self.root.prev
        return self.root.val
    def forward(self, steps):
        """
        :type steps: int
        :rtype: str
        """
        while steps and self.root.next:
            steps -= 1
            self.root = self.root.next
        return self.root.val